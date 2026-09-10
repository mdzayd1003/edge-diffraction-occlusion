"""Markdown and CSV output."""

from __future__ import annotations

import csv
import io
import math
from typing import Any, Mapping, Sequence


def spectrum_markdown(bands: Sequence[Mapping[str, float]], *, maekawa: Sequence[float] | None = None, title: str = "") -> str:
    lines = [f"# {title or 'Insertion loss'}", "", "| Band (Hz) | UTD (dB) |" + (" Maekawa (dB) | Δ |" if maekawa else ""),
             "|---:|---:|" + ("---:|---:|" if maekawa else "")]
    for index, band in enumerate(bands):
        row = f"| {band['centre']:g} | {band['insertion_loss_db']:.2f} |"
        if maekawa:
            row += f" {maekawa[index]:.2f} | {band['insertion_loss_db'] - maekawa[index]:+.2f} |"
        lines.append(row)
    lines.append("")
    return "\n".join(lines)


def study_markdown(results, sensitivity: Mapping[str, Mapping[str, float]], *, centre: float = 500.0) -> str:
    lines = [
        "# Parameter study",
        "",
        f"{len(results)} configurations: "
        f"{len({r.configuration.height for r in results})} barrier heights × "
        f"{len({r.configuration.receiver_distance for r in results})} receiver distances × "
        f"{len({r.configuration.wedge_angle for r in results})} wedge angles.",
        "",
        f"## Marginal means at {centre:g} Hz",
        "",
        "A fully crossed grid, so the mean at each level of a factor is directly",
        "comparable across factors — which a one-at-a-time sweep would not give.",
        "",
    ]
    for factor, levels in sensitivity.items():
        lines += [f"**{factor.replace('_', ' ')}**", "", "| Level | Mean IL (dB) |", "|---|---:|"]
        for level, value in levels.items():
            lines.append(f"| {level} | {value:.2f} |")
        lines.append("")

    ordered = sorted(results, key=lambda r: -r.a_weighted_db)
    lines += [
        "## Extremes, A-weighted",
        "",
        "| Configuration | Height (m) | Distance (m) | Wedge | δ (m) | A-weighted IL (dB) |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for result in ordered[:5] + ordered[-5:]:
        configuration = result.configuration
        lines.append(
            f"| `{configuration.label}` | {configuration.height:g} | {configuration.receiver_distance:g} | "
            f"{math.degrees(configuration.wedge_angle):.0f}° | {result.path_difference_m:.3f} | "
            f"{result.a_weighted_db:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def study_csv(results) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    centres = [band["centre"] for band in results[0].bands] if results else []
    writer.writerow(
        ["label", "height_m", "receiver_distance_m", "wedge_angle_deg", "path_difference_m",
         "fresnel_500hz", "maekawa_500hz_db", "a_weighted_db"] + [f"il_{c:g}hz_db" for c in centres]
    )
    for result in results:
        configuration = result.configuration
        writer.writerow(
            [
                configuration.label,
                f"{configuration.height:g}",
                f"{configuration.receiver_distance:g}",
                f"{math.degrees(configuration.wedge_angle):.0f}",
                f"{result.path_difference_m:.4f}",
                f"{result.fresnel_500hz:.4f}",
                f"{result.maekawa_500hz_db:.3f}",
                f"{result.a_weighted_db:.3f}",
            ]
            + [f"{band['insertion_loss_db']:.3f}" for band in result.bands]
        )
    return buffer.getvalue()


def validation_markdown(result, *, simulated: bool) -> str:
    lines = [
        "# Listening-test validation",
        "",
    ]
    if simulated:
        lines += [
            "> **The ratings behind this table are simulated**, not collected from",
            "> listeners. They exist so the analysis path runs offline. Replace",
            "> `data/listening_test.csv` with real data and nothing else changes.",
            "",
        ]
    lines += [
        f"{result.n_participants} participants × {result.n_stimuli} stimuli.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Spearman ρ, within-participant normalised | **{result.spearman_normalised:.3f}** |",
        f"| 95% bootstrap CI (resampling participants) | [{result.ci_low:.3f}, {result.ci_high:.3f}] |",
        f"| Spearman ρ, raw ratings | {result.spearman_raw:.3f} |",
        f"| Pearson r, normalised | {result.pearson_normalised:.3f} |",
        "",
        "The bootstrap resamples participants rather than individual ratings:",
        f"{result.n_participants} people is the sample size, not "
        f"{result.n_participants * result.n_stimuli} ratings.",
        "",
        "## Per participant",
        "",
        "| Participant | ρ |",
        "|---|---:|",
    ]
    for participant, rho in sorted(result.per_participant.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {participant} | {rho:+.3f} |")
    lines.append("")
    return "\n".join(lines)
