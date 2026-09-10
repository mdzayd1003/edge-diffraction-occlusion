"""The parameter study: 7 barrier heights × 6 receiver distances × 3 wedge angles."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import SPEED_OF_SOUND
from .bands import THIRD_OCTAVE_CENTRES, a_weighted_insertion_loss, insertion_loss_spectrum
from .geometry import Point, Wedge, fresnel_number, maekawa_insertion_loss, path_difference
from .utd import barrier_transfer_function, free_field_transfer_function

BARRIER_HEIGHTS: Tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
RECEIVER_DISTANCES: Tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 60.0, 80.0)
WEDGE_ANGLES: Tuple[float, ...] = (2 * math.pi, 7 * math.pi / 4, 3 * math.pi / 2)

WEDGE_LABELS = {
    2 * math.pi: "thin screen",
    7 * math.pi / 4: "315° wedge",
    3 * math.pi / 2: "270° wedge",
}


@dataclass(frozen=True)
class Configuration:
    height: float
    receiver_distance: float
    wedge_angle: float
    source_distance: float = 5.0
    source_height: float = 1.5
    receiver_height: float = 1.5

    @property
    def wedge(self) -> Wedge:
        return Wedge(open_angle=self.wedge_angle, height=self.height)

    @property
    def source(self) -> Point:
        return Point(-self.source_distance, 0.0, self.source_height)

    @property
    def receiver(self) -> Point:
        return Point(self.receiver_distance, 0.0, self.receiver_height)

    @property
    def label(self) -> str:
        return f"h{self.height:g}-d{self.receiver_distance:g}-w{math.degrees(self.wedge_angle):.0f}"

    def to_dict(self) -> Dict[str, float]:
        payload = asdict(self)
        payload["wedge_label"] = WEDGE_LABELS.get(self.wedge_angle, f"{math.degrees(self.wedge_angle):.0f}°")
        return payload


@dataclass
class Result:
    configuration: Configuration
    bands: List[Dict[str, float]]
    a_weighted_db: float
    path_difference_m: float
    fresnel_500hz: float
    maekawa_500hz_db: float

    def to_dict(self) -> Dict[str, object]:
        return {
            **self.configuration.to_dict(),
            "label": self.configuration.label,
            "a_weighted_db": self.a_weighted_db,
            "path_difference_m": self.path_difference_m,
            "fresnel_500hz": self.fresnel_500hz,
            "maekawa_500hz_db": self.maekawa_500hz_db,
            "bands": self.bands,
        }

    def band_value(self, centre: float) -> float:
        for band in self.bands:
            if band["centre"] == centre:
                return band["insertion_loss_db"]
        raise KeyError(f"no {centre} Hz band in this result")


def grid(
    heights: Sequence[float] = BARRIER_HEIGHTS,
    distances: Sequence[float] = RECEIVER_DISTANCES,
    angles: Sequence[float] = WEDGE_ANGLES,
) -> List[Configuration]:
    return [
        Configuration(height=h, receiver_distance=d, wedge_angle=a)
        for h, d, a in itertools.product(heights, distances, angles)
    ]


def evaluate(
    configuration: Configuration,
    *,
    centres: Sequence[float] = THIRD_OCTAVE_CENTRES,
    c: float = SPEED_OF_SOUND,
    points_per_band: int = 9,
) -> Result:
    wedge = configuration.wedge
    source, receiver = configuration.source, configuration.receiver

    def with_barrier(frequencies):
        return barrier_transfer_function(source, receiver, wedge, frequencies, c=c)

    def without(frequencies):
        return free_field_transfer_function(source, receiver, frequencies, c=c)

    bands = insertion_loss_spectrum(with_barrier, without, centres=centres, points_per_band=points_per_band)
    fresnel = fresnel_number(source, receiver, wedge, 500.0, c=c)
    return Result(
        configuration=configuration,
        bands=[band.to_dict() for band in bands],
        a_weighted_db=a_weighted_insertion_loss(bands),
        path_difference_m=path_difference(source, receiver, wedge),
        fresnel_500hz=fresnel,
        maekawa_500hz_db=maekawa_insertion_loss(fresnel),
    )


def run(
    configurations: Optional[Sequence[Configuration]] = None,
    *,
    centres: Sequence[float] = THIRD_OCTAVE_CENTRES,
    progress: Optional[Callable[[Result], None]] = None,
    **kwargs,
) -> List[Result]:
    results: List[Result] = []
    for configuration in configurations if configurations is not None else grid():
        results.append(evaluate(configuration, centres=centres, **kwargs))
        if progress is not None:
            progress(results[-1])
    return results


def sensitivity(results: Sequence[Result], centre: float = 500.0) -> Dict[str, Dict[str, float]]:
    """How much each factor moves the insertion loss, one factor at a time.

    Reported as the mean band IL at each level of the factor. With a fully
    crossed grid the marginal means are directly comparable, which is the point
    of running the whole cross product rather than a one-at-a-time sweep.
    """
    out: Dict[str, Dict[str, float]] = {"height": {}, "receiver_distance": {}, "wedge_angle": {}}
    for factor in out:
        levels = sorted({getattr(r.configuration, factor) for r in results})
        for level in levels:
            picked = [r.band_value(centre) for r in results if getattr(r.configuration, factor) == level]
            key = WEDGE_LABELS.get(level, f"{level:g}") if factor == "wedge_angle" else f"{level:g}"
            out[factor][key] = float(np.mean(picked))
    return out
