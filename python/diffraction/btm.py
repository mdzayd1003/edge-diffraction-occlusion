"""The Biot-Tolstoy-Medwin edge-diffraction integral, Svensson form.

For a rigid wedge of open angle θ_w (index ν = π/θ_w), with the source at
(r_S, θ_S, z_S) and the receiver at (r_R, θ_R, z_R) in the edge frame, the
first-order diffracted impulse response is a line integral along the edge:

    h(τ) = −(ν c / 4π) ∫ β(z) / (m(z) · l(z) · sinh η(z)) dz

with, at edge position z,

    m = √(r_S² + (z − z_S)²)      l = √(r_R² + (z − z_R)²)
    τ = (m + l) / c
    η = arccosh((m·l + (z − z_S)(z − z_R)) / (r_S r_R))
    β = Σ over the four sign combinations of
            sin(ν φ±±) / (cosh(ν η) − cos(ν φ±±)),  φ±± = π ± θ_S ± θ_R

Two numerical points, both of which are the whole difficulty:

*The integrand is singular at the apex*, where η → 0 and sinh η → 0. The
singularity is integrable. Sampling z on a sinh-spaced grid centred on the apex
concentrates samples exactly where the integrand varies fastest and turns the
singular region into a small number of well-behaved samples.

*The mapping from z to delay is not linear*, so contributions are scattered into
the impulse response with linear interpolation between the two neighbouring
samples, which conserves the integral rather than rounding it to the nearest bin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import SAMPLE_RATE, SPEED_OF_SOUND
from .geometry import (
    EdgeCoordinates,
    Point,
    Wedge,
    apex_position,
    direct_distance,
    is_shadowed,
    shortest_diffracted_path,
    to_edge_coordinates,
)


@dataclass
class ImpulseResponse:
    samples: np.ndarray
    sample_rate: float
    direct_delay: Optional[float] = None
    direct_amplitude: float = 0.0
    apex_delay: float = 0.0

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def energy(self) -> float:
        return float(np.sum(self.samples**2))

    def times(self) -> np.ndarray:
        return np.arange(len(self.samples)) / self.sample_rate


def _beta(nu: float, eta: np.ndarray, theta_s: float, theta_r: float) -> np.ndarray:
    """Svensson's directivity term: the four wedge image contributions."""
    cosh_term = np.cosh(nu * eta)
    total = np.zeros_like(eta)
    for sign_s in (1.0, -1.0):
        for sign_r in (1.0, -1.0):
            phi = math.pi + sign_s * theta_s + sign_r * theta_r
            denominator = cosh_term - math.cos(nu * phi)
            # cosh(νη) ≥ 1 and equals cos(νφ) only in the degenerate case where the
            # receiver sits exactly on a shadow boundary at the apex.
            total += math.sin(nu * phi) / np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    return total


def _sinh_grid(centre: float, half_extent: float, n: int, compression: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """Samples clustered around ``centre``, with their quadrature weights.

    ``z = centre + A·sinh(u)`` for uniform u, so dz = A·cosh(u)·du. Near the apex
    the spacing shrinks like the integrand's variation; far away it grows, which
    is where the integrand is smooth and small.
    """
    if n < 8:
        raise ValueError("need at least 8 edge samples")
    scale = half_extent / math.sinh(compression)
    u = np.linspace(-compression, compression, n)
    du = u[1] - u[0]
    z = centre + scale * np.sinh(u)
    weights = scale * np.cosh(u) * du
    return z, weights


def diffraction_response(
    source: EdgeCoordinates,
    receiver: EdgeCoordinates,
    wedge: Wedge,
    *,
    sample_rate: float = SAMPLE_RATE,
    c: float = SPEED_OF_SOUND,
    n_edge_samples: int = 40001,
    edge_half_length: Optional[float] = None,
    duration: Optional[float] = None,
) -> np.ndarray:
    """The first-order diffracted impulse response, as a sample vector."""
    if source.r <= 0 or receiver.r <= 0:
        raise ValueError("source and receiver must be off the edge itself")

    nu = wedge.index
    apex = apex_position(source, receiver)
    shortest = shortest_diffracted_path(source, receiver)

    # Far along the edge the contribution dies off like 1/(m·l); 40 times the
    # shortest path is well past where it matters at any audible level.
    half_length = edge_half_length if edge_half_length is not None else 40.0 * shortest
    z, weights = _sinh_grid(apex, half_length, n_edge_samples)

    m = np.hypot(source.r, z - source.z)
    l = np.hypot(receiver.r, z - receiver.z)
    delay = (m + l) / c

    argument = (m * l + (z - source.z) * (z - receiver.z)) / (source.r * receiver.r)
    eta = np.arccosh(np.maximum(argument, 1.0 + 1e-15))
    sinh_eta = np.sinh(eta)

    # Units check, which is the easy thing to get wrong here: the integrand is
    # 1/m², dz is m, so ∫ integrand dz is 1/m — the same units as the free-field
    # 1/distance. Multiplying by the sample rate turns that per-bin integral into
    # an impulse-response density, matching how the direct impulse is stored.
    integrand = -(nu / (4 * math.pi)) * _beta(nu, eta, source.theta, receiver.theta) / (m * l * sinh_eta)
    contribution = integrand * weights

    total_duration = duration if duration is not None else shortest / c * 4 + 0.05
    n_samples = int(math.ceil(total_duration * sample_rate))
    response = np.zeros(n_samples, dtype=np.float64)

    # Linear (energy-conserving) scatter into the two neighbouring bins.
    positions = delay * sample_rate
    inside = (positions >= 0) & (positions < n_samples - 1) & np.isfinite(contribution)
    lower = np.floor(positions[inside]).astype(np.int64)
    fraction = positions[inside] - lower
    values = contribution[inside] * sample_rate
    np.add.at(response, lower, values * (1.0 - fraction))
    np.add.at(response, lower + 1, values * fraction)
    return response


def impulse_response(
    source: Point,
    receiver: Point,
    wedge: Wedge,
    *,
    sample_rate: float = SAMPLE_RATE,
    c: float = SPEED_OF_SOUND,
    include_direct: bool = True,
    n_edge_samples: int = 40001,
    duration: Optional[float] = None,
) -> ImpulseResponse:
    """Direct sound (when the receiver can see the source) plus edge diffraction."""
    source_edge = to_edge_coordinates(source, wedge)
    receiver_edge = to_edge_coordinates(receiver, wedge)

    diffracted = diffraction_response(
        source_edge, receiver_edge, wedge, sample_rate=sample_rate, c=c,
        n_edge_samples=n_edge_samples, duration=duration,
    )
    response = ImpulseResponse(
        samples=diffracted,
        sample_rate=sample_rate,
        apex_delay=shortest_diffracted_path(source_edge, receiver_edge) / c,
    )

    shadowed = is_shadowed(source_edge, receiver_edge)
    if include_direct and not shadowed:
        distance = direct_distance(source, receiver)
        amplitude = 1.0 / max(distance, 1e-9)
        position = distance / c * sample_rate
        index = int(math.floor(position))
        if 0 <= index < len(response.samples) - 1:
            fraction = position - index
            response.samples[index] += amplitude * (1 - fraction) * sample_rate
            response.samples[index + 1] += amplitude * fraction * sample_rate
        response.direct_delay = distance / c
        response.direct_amplitude = amplitude
    return response


def free_field_response(
    source: Point,
    receiver: Point,
    *,
    sample_rate: float = SAMPLE_RATE,
    c: float = SPEED_OF_SOUND,
    duration: Optional[float] = None,
    n_samples: Optional[int] = None,
) -> ImpulseResponse:
    """The same source and receiver with no barrier: one delayed impulse."""
    distance = direct_distance(source, receiver)
    amplitude = 1.0 / max(distance, 1e-9)
    if n_samples is None:
        total = duration if duration is not None else distance / c * 4 + 0.05
        n_samples = int(math.ceil(total * sample_rate))
    samples = np.zeros(n_samples, dtype=np.float64)

    position = distance / c * sample_rate
    index = int(math.floor(position))
    if 0 <= index < n_samples - 1:
        fraction = position - index
        samples[index] += amplitude * (1 - fraction) * sample_rate
        samples[index + 1] += amplitude * fraction * sample_rate
    return ImpulseResponse(
        samples=samples, sample_rate=sample_rate, direct_delay=distance / c, direct_amplitude=amplitude
    )
