"""Wedge geometry: world coordinates in, edge-local cylindrical coordinates out.

The scene is a barrier whose top edge runs along the y axis at height ``height``
in the plane x = 0. Source and receiver sit on opposite sides. Everything the
diffraction integral needs is expressed relative to that edge:

``r``  perpendicular distance from the edge
``θ``  angle around the edge, measured **from the source-side face** and
       increasing through the open region, so θ ∈ [0, open_angle] by
       construction. The solid part of the wedge is centred on the downward
       vertical, so a 2π wedge (a thin screen) puts the source side at π/2 and
       the receiver side at 3π/2, while a 270° wedge tilts both faces up by 45°.
       Anything outside [0, open_angle] is inside the body of the wedge.
``z``  position along the edge
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

TWO_PI = 2 * math.pi


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass(frozen=True)
class EdgeCoordinates:
    """One endpoint expressed relative to the edge."""

    r: float
    theta: float
    z: float


@dataclass(frozen=True)
class Wedge:
    """A rigid wedge. ``open_angle`` is the angle of the region sound lives in.

    2π is a thin screen (the classic barrier). π is a quarter-space corner, e.g.
    the top of a thick wall. The wedge index ν = π / open_angle appears
    throughout the diffraction integral.
    """

    open_angle: float = TWO_PI
    height: float = 3.0

    def __post_init__(self) -> None:
        if not 0 < self.open_angle <= TWO_PI:
            raise ValueError(f"open angle must lie in (0, 2π], got {self.open_angle}")

    @property
    def index(self) -> float:
        return math.pi / self.open_angle

    @property
    def half_solid_angle(self) -> float:
        """How far each face is tilted up from the downward vertical."""
        return (TWO_PI - self.open_angle) / 2.0


def to_edge_coordinates(point: Point, wedge: Wedge) -> EdgeCoordinates:
    """Express a world point in the edge's cylindrical frame."""
    dx = point.x
    dz = point.z - wedge.height
    r = math.hypot(dx, dz)
    # Straight down is 0 before the shift; the angle grows through the source side
    # (negative x), over the top, and round to the receiver side. Subtracting the
    # face tilt puts θ = 0 on the source-side face for any wedge angle.
    theta = (math.atan2(-dx, -dz) - wedge.half_solid_angle) % TWO_PI
    return EdgeCoordinates(r=r, theta=theta, z=point.y)


def is_shadowed(source: EdgeCoordinates, receiver: EdgeCoordinates) -> bool:
    """True when the straight line from source to receiver crosses the wedge.

    The shadow boundary sits at an angular separation of exactly π. The test uses
    the unsigned separation, not a difference taken modulo 2π: the modulo version
    is not symmetric under swapping source and receiver, and an asymmetric shadow
    test breaks reciprocity in whatever is built on top of it.
    """
    return abs(receiver.theta - source.theta) > math.pi


def direct_distance(source: Point, receiver: Point) -> float:
    return float(np.linalg.norm(source.as_array() - receiver.as_array()))


def apex_position(source: EdgeCoordinates, receiver: EdgeCoordinates) -> float:
    """Where along the edge the shortest diffracted path crosses.

    The unfolded path length r_S + r_R is minimised by splitting the along-edge
    separation in proportion to the two perpendicular distances.
    """
    total = source.r + receiver.r
    if total <= 0:
        return source.z
    return float(source.z + (receiver.z - source.z) * source.r / total)


def shortest_diffracted_path(source: EdgeCoordinates, receiver: EdgeCoordinates) -> float:
    z = apex_position(source, receiver)
    return math.hypot(source.r, z - source.z) + math.hypot(receiver.r, z - receiver.z)


def path_difference(source: Point, receiver: Point, wedge: Wedge) -> float:
    """δ = (shortest path over the edge) − (straight-line distance).

    The single number the classical barrier formulae are written in terms of.
    """
    s = to_edge_coordinates(source, wedge)
    r = to_edge_coordinates(receiver, wedge)
    return shortest_diffracted_path(s, r) - direct_distance(source, receiver)


def fresnel_number(source: Point, receiver: Point, wedge: Wedge, frequency: float, c: float = 343.0) -> float:
    """N = 2δ / λ. Positive in the shadow, negative in the illuminated zone."""
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    delta = path_difference(source, receiver, wedge)
    s = to_edge_coordinates(source, wedge)
    r = to_edge_coordinates(receiver, wedge)
    sign = 1.0 if is_shadowed(s, r) else -1.0
    return sign * 2.0 * abs(delta) * frequency / c


def maekawa_insertion_loss(fresnel: float) -> float:
    """Maekawa's empirical barrier chart, as a closed form.

    An independent reference: it comes from measurements rather than from the
    diffraction integral, so agreement between the two is worth something. Used
    in the tests to check the BTM implementation, never as a substitute for it.
    """
    if fresnel > 0:
        argument = math.sqrt(2 * math.pi * fresnel)
        return 5.0 + 20.0 * math.log10(argument / math.tanh(argument))
    if fresnel == 0:
        return 5.0
    argument = math.sqrt(2 * math.pi * abs(fresnel))
    if argument >= math.pi / 2:  # beyond the chart's validity: no attenuation
        return 0.0
    return max(0.0, 5.0 - 20.0 * math.log10(math.tan(argument) / argument))
