"""Comparing the simulation against listening-test ratings.

Three decisions that matter more than the correlation coefficient itself:

**Ratings are normalised within participant.** People use a 1-7 scale
differently — one person lives in 3-5, another swings 1-7 — and that is
individual scale use, not disagreement about the stimuli. Z-scoring each
participant's ratings removes it. The raw correlation is reported alongside so
the effect of doing it is visible.

**Spearman, not Pearson.** The simulation predicts insertion loss in dB; the
listener gives an ordinal rating. Only the ordering is comparable, and assuming
the rating scale is linear in dB would be assuming the answer.

**The bootstrap resamples participants, not ratings.** Ten participants is the
sample size; resampling the 120 individual ratings would treat them as 120
independent observations and produce an interval several times too narrow.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Ratings:
    participants: List[str]
    stimuli: List[str]
    matrix: np.ndarray  # (participants, stimuli)

    def __post_init__(self) -> None:
        if self.matrix.shape != (len(self.participants), len(self.stimuli)):
            raise ValueError("ratings matrix does not match its labels")

    @property
    def n_participants(self) -> int:
        return len(self.participants)

    @property
    def n_stimuli(self) -> int:
        return len(self.stimuli)


def load_ratings(path: str | Path) -> Ratings:
    """Read long-format CSV: participant, stimulus, rating."""
    rows = list(csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError(f"{path} contains no ratings")
    missing = {"participant", "stimulus", "rating"} - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")

    participants = sorted({row["participant"] for row in rows})
    stimuli = sorted({row["stimulus"] for row in rows})
    matrix = np.full((len(participants), len(stimuli)), np.nan)
    for row in rows:
        matrix[participants.index(row["participant"]), stimuli.index(row["stimulus"])] = float(row["rating"])
    if np.isnan(matrix).any():
        gaps = int(np.isnan(matrix).sum())
        raise ValueError(f"{path} is not a complete design: {gaps} participant/stimulus cells are missing")
    return Ratings(participants=participants, stimuli=stimuli, matrix=matrix)


def normalise_within_participant(ratings: Ratings) -> np.ndarray:
    """Z-score each participant's row. A flat rater becomes all zeros, not NaN."""
    matrix = ratings.matrix.astype(np.float64)
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (matrix - mean) / std


def rank(values: Sequence[float]) -> np.ndarray:
    """Average ranks, so ties do not bias the correlation."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)

    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError("both sequences must be the same length")
    if len(x) < 3:
        raise ValueError("need at least three points for a correlation")
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y / denominator) if denominator > 0 else 0.0


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(rank(x), rank(y))


@dataclass
class ValidationResult:
    spearman_normalised: float
    spearman_raw: float
    pearson_normalised: float
    ci_low: float
    ci_high: float
    n_participants: int
    n_stimuli: int
    per_participant: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "spearman_normalised": self.spearman_normalised,
            "spearman_raw": self.spearman_raw,
            "pearson_normalised": self.pearson_normalised,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_participants": self.n_participants,
            "n_stimuli": self.n_stimuli,
            "per_participant": self.per_participant,
        }


def bootstrap_participants(
    normalised: np.ndarray, predicted: Sequence[float], *, resamples: int = 5000, seed: int = 0, alpha: float = 0.05
) -> Tuple[float, float]:
    """Percentile bootstrap over participants — the unit of independence."""
    rng = np.random.default_rng(seed)
    n = normalised.shape[0]
    values: List[float] = []
    for _ in range(resamples):
        index = rng.integers(0, n, size=n)
        mean_rating = normalised[index].mean(axis=0)
        try:
            values.append(spearman(mean_rating, predicted))
        except ValueError:  # pragma: no cover - needs a degenerate resample
            continue
    if not values:  # pragma: no cover
        return (float("nan"), float("nan"))
    values.sort()
    return (
        float(values[int((alpha / 2) * (len(values) - 1))]),
        float(values[int((1 - alpha / 2) * (len(values) - 1))]),
    )


def validate(
    ratings: Ratings, predictions: Dict[str, float], *, resamples: int = 5000, seed: int = 0
) -> ValidationResult:
    missing = [s for s in ratings.stimuli if s not in predictions]
    if missing:
        raise KeyError(f"no prediction for stimulus/stimuli: {missing}")

    predicted = [predictions[s] for s in ratings.stimuli]
    normalised = normalise_within_participant(ratings)

    low, high = bootstrap_participants(normalised, predicted, resamples=resamples, seed=seed)
    return ValidationResult(
        spearman_normalised=spearman(normalised.mean(axis=0), predicted),
        spearman_raw=spearman(ratings.matrix.mean(axis=0), predicted),
        pearson_normalised=pearson(normalised.mean(axis=0), predicted),
        ci_low=low,
        ci_high=high,
        n_participants=ratings.n_participants,
        n_stimuli=ratings.n_stimuli,
        per_participant={
            name: spearman(normalised[i], predicted) for i, name in enumerate(ratings.participants)
        },
    )


def simulate_ratings(
    predictions: Dict[str, float],
    *,
    n_participants: int = 10,
    seed: int = 0,
    noise: float = 0.8,
    scale_range: Tuple[float, float] = (0.6, 1.4),
    steps: int = 7,
) -> Ratings:
    """Generate a stand-in dataset. **Not** a real listening test.

    Each simulated participant has their own offset and their own use of the
    scale, then rates a monotone function of the predicted insertion loss with
    noise on top. It exists so the analysis path runs and is tested offline; a
    real study replaces the CSV and nothing else changes.
    """
    rng = np.random.default_rng(seed)
    stimuli = sorted(predictions)
    values = np.array([predictions[s] for s in stimuli], dtype=np.float64)
    spread = values.max() - values.min()
    centred = (values - values.mean()) / (spread if spread > 0 else 1.0)

    matrix = np.zeros((n_participants, len(stimuli)))
    for participant in range(n_participants):
        offset = rng.normal(0.0, 0.6)
        scale = rng.uniform(*scale_range)
        raw = (steps + 1) / 2 + scale * centred * (steps - 1) + offset + rng.normal(0.0, noise, size=len(stimuli))
        matrix[participant] = np.clip(np.round(raw), 1, steps)

    return Ratings(
        participants=[f"P{i + 1:02d}" for i in range(n_participants)], stimuli=stimuli, matrix=matrix
    )


def provenance_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".provenance.json")


def read_provenance(path: str | Path) -> Dict[str, object]:
    """What produced this ratings file. Absent means 'unknown', not 'real'."""
    import json

    sidecar = provenance_path(path)
    if not sidecar.exists():
        return {"source": "unknown"}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def write_ratings(ratings: Ratings, path: str | Path, *, provenance: Optional[Dict[str, object]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["participant", "stimulus", "rating"])
        for i, participant in enumerate(ratings.participants):
            for j, stimulus in enumerate(ratings.stimuli):
                writer.writerow([participant, stimulus, int(ratings.matrix[i, j])])

    import json

    provenance_path(path).write_text(
        json.dumps(provenance or {"source": "simulated"}, indent=2), encoding="utf-8"
    )
    return path
