"""Figures: band insertion loss, the parameter sweeps, the listening comparison."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def band_spectrum(bands: Sequence[Mapping[str, float]], path: str | Path, *, reference=None, title: str = "") -> Path:
    centres = [b["centre"] for b in bands]
    values = [b["insertion_loss_db"] for b in bands]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(centres, values, marker="o", markersize=3, label="UTD")
    if reference is not None:
        ax.semilogx(centres, list(reference), linestyle="--", color="#8C8C8C", label="Maekawa")
        ax.legend()
    ax.set_xlabel("third-octave band centre (Hz)")
    ax.set_ylabel("insertion loss (dB)")
    ax.set_title(title or "Insertion loss")
    ax.grid(which="both", alpha=0.25)
    return _save(fig, path)


def parameter_surface(
    results, path: str | Path, *, centre: float = 500.0, wedge_angle: float = 2 * math.pi
) -> Path:
    picked = [r for r in results if r.configuration.wedge_angle == wedge_angle]
    heights = sorted({r.configuration.height for r in picked})
    distances = sorted({r.configuration.receiver_distance for r in picked})
    values = np.full((len(heights), len(distances)), np.nan)
    for result in picked:
        i = heights.index(result.configuration.height)
        j = distances.index(result.configuration.receiver_distance)
        values[i, j] = result.band_value(centre)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    image = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(distances)), [f"{d:g}" for d in distances])
    ax.set_yticks(range(len(heights)), [f"{h:g}" for h in heights])
    for i in range(len(heights)):
        for j in range(len(distances)):
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", color="white", fontsize=7)
    ax.set_xlabel("receiver distance (m)")
    ax.set_ylabel("barrier height (m)")
    ax.set_title(f"Insertion loss at {centre:g} Hz — {math.degrees(wedge_angle):.0f}° wedge")
    fig.colorbar(image, ax=ax, label="dB")
    return _save(fig, path)


def sensitivity_bars(sensitivity: Mapping[str, Mapping[str, float]], path: str | Path) -> Path:
    fig, axes = plt.subplots(1, len(sensitivity), figsize=(4 * len(sensitivity), 3.5))
    axes = np.atleast_1d(axes)
    for ax, (factor, levels) in zip(axes, sensitivity.items()):
        ax.bar(list(levels), list(levels.values()), color="#4C72B0")
        ax.set_title(factor.replace("_", " "))
        ax.set_ylabel("mean IL at 500 Hz (dB)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def impulse_response_plot(samples: np.ndarray, sample_rate: float, path: str | Path, *, title: str = "") -> Path:
    times = np.arange(len(samples)) / sample_rate * 1000.0
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(times, samples, linewidth=0.8)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("amplitude")
    ax.set_title(title or "Diffracted impulse response")
    ax.grid(alpha=0.25)
    peak = int(np.argmax(np.abs(samples)))
    ax.set_xlim(max(0.0, times[peak] - 5), times[peak] + 25)
    return _save(fig, path)


def listening_scatter(
    predicted: Sequence[float], normalised: np.ndarray, labels: Sequence[str], path: str | Path, *, rho: float = 0.0
) -> Path:
    mean = normalised.mean(axis=0)
    error = normalised.std(axis=0) / math.sqrt(max(normalised.shape[0], 1))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(predicted, mean, yerr=error, fmt="o", capsize=3, color="#4C72B0")
    for x, y, label in zip(predicted, mean, labels):
        ax.annotate(label, (x, y), fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("predicted A-weighted insertion loss (dB)")
    ax.set_ylabel("mean normalised rating (z)")
    ax.set_title(f"Simulation against listener ratings (Spearman ρ = {rho:.2f})")
    ax.grid(alpha=0.25)
    return _save(fig, path)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
