"""Third-octave bands, band insertion loss, and the time-domain response.

Insertion loss is computed by integrating |H(f)|² over each band rather than
sampling the band centre. Diffraction is smooth in frequency so the difference is
small, but it is not zero near a shadow boundary where the response has structure
inside a band, and integrating costs nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import SAMPLE_RATE, SPEED_OF_SOUND

# numpy renamed trapz -> trapezoid in 2.0, and deprecated the old spelling. Bind
# the one that exists so this runs on either: pinning to the new name alone made
# the module quietly numpy-2-only while requirements.txt still claimed >=1.24.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# The standard nominal third-octave centres, 50 Hz to 10 kHz.
THIRD_OCTAVE_CENTRES: Tuple[float, ...] = (
    50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
)

_RATIO = 2.0 ** (1.0 / 6.0)  # half a third-octave


def band_edges(centre: float) -> Tuple[float, float]:
    if centre <= 0:
        raise ValueError("band centre must be positive")
    return centre / _RATIO, centre * _RATIO


def band_frequencies(centre: float, n: int = 9) -> np.ndarray:
    """Log-spaced sample points across one band, for the energy integral."""
    if n < 2:
        raise ValueError("need at least two points per band")
    low, high = band_edges(centre)
    return np.geomspace(low, high, n)


@dataclass
class BandInsertionLoss:
    centre: float
    insertion_loss_db: float
    barrier_level_db: float
    free_level_db: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "centre": self.centre,
            "insertion_loss_db": self.insertion_loss_db,
            "barrier_level_db": self.barrier_level_db,
            "free_level_db": self.free_level_db,
        }


def band_energy(frequencies: np.ndarray, response: np.ndarray) -> float:
    """Mean-square level across a band, integrated on a log-frequency axis."""
    magnitude_squared = np.abs(np.asarray(response)) ** 2
    log_f = np.log(np.asarray(frequencies, dtype=np.float64))
    if len(log_f) < 2:
        return float(magnitude_squared.mean())
    return float(_trapezoid(magnitude_squared, log_f) / (log_f[-1] - log_f[0]))


def insertion_loss_spectrum(
    barrier, free_field, *, centres: Sequence[float] = THIRD_OCTAVE_CENTRES, points_per_band: int = 9
) -> List[BandInsertionLoss]:
    """``barrier`` and ``free_field`` are callables f -> complex response."""
    out: List[BandInsertionLoss] = []
    for centre in centres:
        frequencies = band_frequencies(centre, points_per_band)
        with_barrier = band_energy(frequencies, barrier(frequencies))
        without = band_energy(frequencies, free_field(frequencies))
        if without <= 0:
            raise ValueError(f"free-field energy is zero in the {centre} Hz band")
        out.append(
            BandInsertionLoss(
                centre=float(centre),
                insertion_loss_db=float(10.0 * math.log10(without / max(with_barrier, 1e-30))),
                barrier_level_db=float(10.0 * math.log10(max(with_barrier, 1e-30))),
                free_level_db=float(10.0 * math.log10(without)),
            )
        )
    return out


def a_weighting_db(frequencies: np.ndarray) -> np.ndarray:
    """IEC 61672 A-weighting, for the single-figure summary."""
    f = np.asarray(frequencies, dtype=np.float64)
    f2 = f**2
    numerator = (12194.0**2) * f2**2
    denominator = (
        (f2 + 20.6**2)
        * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
        * (f2 + 12194.0**2)
    )
    return 20.0 * np.log10(numerator / denominator) + 2.00


def a_weighted_insertion_loss(bands: Sequence[BandInsertionLoss]) -> float:
    """One number: the A-weighted difference, assuming a flat source spectrum."""
    centres = np.array([b.centre for b in bands], dtype=np.float64)
    weights = 10.0 ** (a_weighting_db(centres) / 10.0)
    free = weights * 10.0 ** (np.array([b.free_level_db for b in bands]) / 10.0)
    with_barrier = weights * 10.0 ** (np.array([b.barrier_level_db for b in bands]) / 10.0)
    return float(10.0 * math.log10(free.sum() / max(with_barrier.sum(), 1e-30)))


def impulse_response(
    transfer_function,
    *,
    sample_rate: float = SAMPLE_RATE,
    n_fft: int = 8192,
    low_cut: float = 20.0,
) -> np.ndarray:
    """Time-domain response by inverse FFT of the transfer function.

    Below ``low_cut`` the diffraction coefficient's 1/√k factor sends the response
    to infinity as f → 0, so the band is rolled off rather than evaluated: the
    theory has nothing to say there and neither does the impulse response.
    """
    if n_fft % 2:
        raise ValueError("n_fft must be even")
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    spectrum = np.zeros(len(frequencies), dtype=np.complex128)

    usable = frequencies >= low_cut
    spectrum[usable] = transfer_function(frequencies[usable])
    # A raised-cosine ramp over the octave below low_cut, so the cut is not a step.
    ramp = (frequencies > low_cut / 2) & (frequencies < low_cut)
    if ramp.any():
        fraction = (frequencies[ramp] - low_cut / 2) / (low_cut / 2)
        spectrum[ramp] = transfer_function(frequencies[ramp]) * 0.5 * (1 - np.cos(math.pi * fraction))
    return np.fft.irfft(spectrum, n=n_fft)
