"""Wedge diffraction by the Uniform Theory of Diffraction (Kouyoumjian & Pathak).

The diffracted field from a rigid wedge of exterior angle nπ, for a point source:

    p_d = D(φ, φ′) · e^{−jk(s+s′)} / √(s s′ (s + s′))

    D = −e^{−jπ/4} / (2n √(2πk) sin β₀) · Σ_{β ∈ {φ−φ′, φ+φ′}} Σ_{±}
            cot((π ± β) / 2n) · F(k L a^±(β))

with L = s s′ sin²β₀ / (s + s′), a^±(β) = 2cos²((2πnN^± − β)/2), and F the UTD
transition function. The four terms are the two shadow boundaries and the two
reflection boundaries; the transition function is what makes the result finite
and continuous across all of them, which the geometrical-optics diffraction
coefficient is not.

Two things make this implementation checkable rather than merely plausible, and
both are tests:

* At a shadow boundary the total field must be exactly **half** the free field —
  the diffracted term steps in to cover the discontinuity as the direct sound
  switches off. Nothing is fitted to make that happen; it falls out of F.
* Well into the shadow it must land within a few dB of Maekawa's chart, which
  comes from measurements rather than from this theory.

Near a boundary the cotangent is singular and F is zero. The product is finite,
and its limit is used directly instead of multiplying an infinity by a zero.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
from scipy.special import fresnel

from . import SPEED_OF_SOUND
from .geometry import Point, Wedge, apex_position, direct_distance, is_shadowed, to_edge_coordinates

BOUNDARY_TOLERANCE = 1e-6


def transition_function(x: np.ndarray) -> np.ndarray:
    """F(X) = 2j √X e^{jX} ∫_√X^∞ e^{−jτ²} dτ.

    F(X) → 1 for X ≫ 1 (geometrical optics recovered) and → 0 as X → 0 (the
    boundary, where the cotangent blows up to meet it).
    """
    x = np.asarray(x, dtype=np.float64)
    argument = np.sqrt(np.maximum(2.0 * x / math.pi, 0.0))
    sin_part, cos_part = fresnel(argument)
    tail = math.sqrt(math.pi / 2.0) * ((0.5 - cos_part) - 1j * (0.5 - sin_part))
    return 2j * np.sqrt(np.maximum(x, 0.0)) * np.exp(1j * x) * tail


def _nearest_n(beta: float, n: float, sign: int) -> float:
    target = (beta + sign * math.pi) / (2.0 * math.pi * n)
    return float(round(target))


def _cot_times_f(beta: float, sign: int, n: float, k_l: np.ndarray) -> np.ndarray:
    """One of the four terms: cot((π + sign·β)/2n) · F(kL a^{sign}(β))."""
    integer = _nearest_n(beta, n, sign)
    angle = (2.0 * math.pi * n * integer - beta) / 2.0
    a = 2.0 * math.cos(angle) ** 2

    cot_argument = (math.pi + sign * beta) / (2.0 * n)
    sin_cot = math.sin(cot_argument)

    if abs(sin_cot) > BOUNDARY_TOLERANCE:
        return (math.cos(cot_argument) / sin_cot) * transition_function(k_l * a)

    # On a boundary: cot → ∞ and F → 0. Their product has the closed-form limit
    # below (Kouyoumjian & Pathak), which is what keeps the field continuous.
    epsilon = beta - (2.0 * math.pi * n * integer - sign * math.pi)
    direction = 1.0 if epsilon >= 0 else -1.0
    return n * (np.sqrt(2.0 * math.pi * k_l) * direction - 2.0 * k_l * epsilon * np.exp(1j * math.pi / 4.0)) * np.exp(
        1j * math.pi / 4.0
    )


def diffraction_coefficient(
    phi_source: float,
    phi_receiver: float,
    wedge_index_n: float,
    k: np.ndarray,
    l: np.ndarray,
    *,
    beta_0: float = math.pi / 2,
) -> np.ndarray:
    """The UTD coefficient D for a rigid wedge, at each wavenumber in ``k``."""
    k = np.asarray(k, dtype=np.float64)
    k_l = k * np.asarray(l, dtype=np.float64)

    total = np.zeros_like(k, dtype=np.complex128)
    for beta in (phi_receiver - phi_source, phi_receiver + phi_source):
        for sign in (1, -1):
            total = total + _cot_times_f(beta, sign, wedge_index_n, k_l)

    prefactor = -np.exp(-1j * math.pi / 4.0) / (2.0 * wedge_index_n * np.sqrt(2.0 * math.pi * k) * math.sin(beta_0))
    return prefactor * total


def _faces(wedge: Wedge, theta: float) -> float:
    """Angle measured from the wedge's lower face, which UTD calls the o-face.

    ``geometry`` measures θ from straight down; for a wedge whose open region is
    ``open_angle``, the lower face sits at θ = 0 by construction, so the two
    conventions already agree.
    """
    return theta


def barrier_transfer_function(
    source: Point,
    receiver: Point,
    wedge: Wedge,
    frequencies: Sequence[float] | np.ndarray,
    *,
    c: float = SPEED_OF_SOUND,
    include_direct: bool = True,
) -> np.ndarray:
    """Total pressure at the receiver, per unit free-field pressure at 1 m.

    Direct sound is included when the receiver can see the source; the diffracted
    term is always included.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if np.any(frequencies <= 0):
        raise ValueError("frequencies must be positive; DC has no UTD solution")

    source_edge = to_edge_coordinates(source, wedge)
    receiver_edge = to_edge_coordinates(receiver, wedge)
    if source_edge.r <= 0 or receiver_edge.r <= 0:
        raise ValueError("source and receiver must be off the edge itself")
    for name, point in (("source", source_edge), ("receiver", receiver_edge)):
        if point.theta > wedge.open_angle + 1e-9:
            raise ValueError(
                f"{name} lies at θ = {math.degrees(point.theta):.1f}°, inside the body of a "
                f"{math.degrees(wedge.open_angle):.1f}° wedge"
            )

    # Skew incidence: the diffraction point is where Keller's cone condition
    # holds, i.e. the ray makes the same angle with the edge on both sides. That
    # is the apex of the shortest path over the edge, and splitting the along-edge
    # offset there is what makes the result reciprocal — putting all of the offset
    # on one leg is not, and shows up immediately when source and receiver swap.
    apex = apex_position(source_edge, receiver_edge)
    s_prime = math.hypot(source_edge.r, apex - source_edge.z)
    s = math.hypot(receiver_edge.r, apex - receiver_edge.z)
    beta_0 = math.asin(min(1.0, source_edge.r / s_prime))

    k = 2.0 * math.pi * frequencies / c
    n = wedge.open_angle / math.pi
    l = s * s_prime * math.sin(beta_0) ** 2 / (s + s_prime)

    coefficient = diffraction_coefficient(
        _faces(wedge, source_edge.theta), _faces(wedge, receiver_edge.theta), n, k,
        np.full_like(k, l), beta_0=beta_0,
    )
    diffracted = coefficient * np.exp(-1j * k * (s + s_prime)) / math.sqrt(s * s_prime * (s + s_prime))

    if include_direct and not is_shadowed(source_edge, receiver_edge):
        distance = direct_distance(source, receiver)
        diffracted = diffracted + np.exp(-1j * k * distance) / distance
    return diffracted


def free_field_transfer_function(
    source: Point, receiver: Point, frequencies: Sequence[float] | np.ndarray, *, c: float = SPEED_OF_SOUND
) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=np.float64)
    distance = direct_distance(source, receiver)
    k = 2.0 * math.pi * frequencies / c
    return np.exp(-1j * k * distance) / distance
