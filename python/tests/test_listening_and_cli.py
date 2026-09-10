import json
import math
from pathlib import Path

import numpy as np
import pytest

from diffraction.__main__ import STIMULI, main
from diffraction.listening import (
    Ratings,
    bootstrap_participants,
    load_ratings,
    normalise_within_participant,
    pearson,
    provenance_path,
    rank,
    read_provenance,
    simulate_ratings,
    spearman,
    validate,
    write_ratings,
)


@pytest.fixture()
def predictions():
    return {f"s{i}": float(i) for i in range(12)}


@pytest.fixture()
def ratings(predictions):
    return simulate_ratings(predictions, n_participants=10, seed=0)


def test_ranking_averages_ties():
    assert rank([10, 20, 20, 30]).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_ranking_is_one_based_and_covers_every_position():
    assert sorted(rank([5, 3, 9, 1]).tolist()) == [1.0, 2.0, 3.0, 4.0]


def test_spearman_matches_scipy_including_ties():
    from scipy.stats import spearmanr

    x = [1, 2, 2, 3, 5, 8]
    y = [2, 1, 4, 4, 7, 6]
    assert spearman(x, y) == pytest.approx(float(spearmanr(x, y).statistic))


def test_spearman_is_one_for_any_monotone_relationship():
    x = [1, 2, 3, 4, 5]
    assert spearman(x, [v**3 for v in x]) == pytest.approx(1.0)
    assert spearman(x, [-v for v in x]) == pytest.approx(-1.0)


def test_pearson_would_not_give_one_on_that_relationship():
    """Which is the reason the analysis uses Spearman: dB against an ordinal scale."""
    x = [1, 2, 3, 4, 5]
    assert pearson(x, [v**3 for v in x]) < 0.95


def test_a_correlation_needs_at_least_three_points():
    with pytest.raises(ValueError):
        pearson([1, 2], [1, 2])


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        pearson([1, 2, 3], [1, 2])


def test_normalisation_gives_every_participant_the_same_scale(ratings):
    normalised = normalise_within_participant(ratings)
    np.testing.assert_allclose(normalised.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(normalised.std(axis=1), 1.0, atol=1e-12)


def test_a_participant_who_gave_one_rating_to_everything_becomes_zeros():
    flat = Ratings(participants=["P1"], stimuli=["a", "b", "c"], matrix=np.full((1, 3), 4.0))
    assert np.allclose(normalise_within_participant(flat), 0.0)


def test_normalisation_cannot_change_a_participants_ordering(ratings):
    normalised = normalise_within_participant(ratings)
    for raw_row, normalised_row in zip(ratings.matrix, normalised):
        np.testing.assert_array_equal(np.argsort(raw_row, kind="stable"), np.argsort(normalised_row, kind="stable"))


def test_the_bootstrap_resamples_participants_not_ratings(predictions, ratings):
    """Ten people, not 120 ratings — the interval must reflect the smaller n."""
    normalised = normalise_within_participant(ratings)
    predicted = [predictions[s] for s in ratings.stimuli]

    low, high = bootstrap_participants(normalised, predicted, resamples=800, seed=0)
    point = spearman(normalised.mean(axis=0), predicted)
    assert low <= point <= high
    assert high - low > 0.0


def test_the_bootstrap_is_reproducible(predictions, ratings):
    normalised = normalise_within_participant(ratings)
    predicted = [predictions[s] for s in ratings.stimuli]
    first = bootstrap_participants(normalised, predicted, resamples=300, seed=7)
    second = bootstrap_participants(normalised, predicted, resamples=300, seed=7)
    assert first == second


def test_validation_recovers_the_relationship_the_simulation_was_built_with(predictions, ratings):
    result = validate(ratings, predictions, resamples=500)
    assert result.spearman_normalised > 0.8
    assert result.n_participants == 10 and result.n_stimuli == 12
    assert len(result.per_participant) == 10


def test_validation_refuses_a_stimulus_it_cannot_predict(ratings):
    with pytest.raises(KeyError, match="no prediction"):
        validate(ratings, {"s0": 1.0}, resamples=10)


def test_ratings_round_trip_through_csv_with_their_provenance(tmp_path, ratings):
    path = write_ratings(ratings, tmp_path / "r.csv", provenance={"source": "simulated", "seed": 0})
    reloaded = load_ratings(path)
    assert reloaded.participants == ratings.participants
    assert reloaded.stimuli == ratings.stimuli
    np.testing.assert_array_equal(reloaded.matrix, ratings.matrix)
    assert read_provenance(path)["source"] == "simulated"
    assert provenance_path(path).exists()


def test_an_unknown_provenance_is_not_treated_as_real(tmp_path, ratings):
    path = tmp_path / "bare.csv"
    write_ratings(ratings, path)
    provenance_path(path).unlink()
    assert read_provenance(path)["source"] == "unknown"


def test_an_incomplete_design_is_refused(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("participant,stimulus,rating\nP1,a,3\nP1,b,4\nP2,a,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a complete design"):
        load_ratings(path)


def test_a_missing_column_is_refused(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("participant,rating\nP1,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        load_ratings(path)


def test_an_empty_file_is_refused(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("participant,stimulus,rating\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no ratings"):
        load_ratings(path)


def test_the_stimulus_set_is_a_subset_of_the_study_grid():
    from diffraction.study import grid

    labels = {c.label for c in grid()}
    assert {c.label for c in STIMULI} <= labels
    assert len(STIMULI) == 12


def test_the_spectrum_command_writes_its_outputs(tmp_path, capsys):
    results = tmp_path / "run"
    assert main(["--results", str(results), "spectrum", "--height", "3", "--receiver-distance", "10"]) == 0
    payload = json.loads((results / "spectrum.json").read_text())
    assert len(payload["bands"]) == 24
    assert payload["a_weighted_db"] > 0
    for name in ("spectrum.md", "spectrum.png", "impulse_response.png", "impulse_response.npy"):
        assert (results / name).exists(), name


def test_the_study_command_writes_a_row_per_configuration(tmp_path, capsys):
    results = tmp_path / "run"
    assert main(["--results", str(results), "study"]) == 0
    rows = (results / "study.csv").read_text().strip().splitlines()
    assert len(rows) == 127, "a header plus 126 configurations"
    payload = json.loads((results / "study.json").read_text())
    assert set(payload["sensitivity"]) == {"height", "receiver_distance", "wedge_angle"}
    for name in ("study.md", "surface_thin_screen.png", "sensitivity.png"):
        assert (results / name).exists(), name


def test_simulate_then_validate_round_trips(tmp_path, capsys):
    results = tmp_path / "run"
    ratings_path = tmp_path / "ratings.csv"
    assert main(["--results", str(results), "simulate-ratings", "--out", str(ratings_path)]) == 0
    assert "NOT real listener ratings" in capsys.readouterr().out

    assert main(["--results", str(results), "validate", "--ratings", str(ratings_path), "--resamples", "300"]) == 0
    markdown = (results / "validation.md").read_text()
    assert "simulated" in markdown, "a simulated dataset must say so in the report"
    payload = json.loads((results / "validation.json").read_text())
    assert payload["n_participants"] == 10
    assert payload["ci_low"] <= payload["spearman_normalised"] <= payload["ci_high"]


def test_validating_against_a_missing_file_says_what_to_do(tmp_path):
    with pytest.raises(SystemExit, match="simulate-ratings"):
        main(["--results", str(tmp_path), "validate", "--ratings", str(tmp_path / "nope.csv")])


def test_the_shipped_ratings_are_marked_simulated():
    path = Path(__file__).resolve().parents[2] / "data" / "listening_test.csv"
    assert path.exists()
    assert read_provenance(path)["source"] == "simulated"
