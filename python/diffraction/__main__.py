"""CLI: ``spectrum`` one geometry, ``study`` the grid, ``validate`` against ratings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import SAMPLE_RATE, SPEED_OF_SOUND
from .bands import THIRD_OCTAVE_CENTRES, a_weighted_insertion_loss, impulse_response, insertion_loss_spectrum
from .geometry import Point, Wedge, fresnel_number, maekawa_insertion_loss, path_difference
from .listening import (
    load_ratings,
    normalise_within_participant,
    read_provenance,
    simulate_ratings,
    validate as validate_ratings,
    write_ratings,
)
from .plots import band_spectrum, impulse_response_plot, listening_scatter, parameter_surface, sensitivity_bars
from .report import spectrum_markdown, study_csv, study_markdown, validation_markdown
from .study import Configuration, evaluate, grid, run, sensitivity
from .utd import barrier_transfer_function, free_field_transfer_function

STIMULI = [
    Configuration(height=h, receiver_distance=d, wedge_angle=a)
    for h, d, a in [
        (2.0, 10.0, 2 * math.pi), (2.0, 40.0, 2 * math.pi),
        (3.0, 10.0, 2 * math.pi), (3.0, 40.0, 2 * math.pi),
        (4.0, 10.0, 2 * math.pi), (4.0, 40.0, 2 * math.pi),
        (6.0, 10.0, 2 * math.pi), (6.0, 40.0, 2 * math.pi),
        (3.0, 10.0, 3 * math.pi / 2), (3.0, 40.0, 3 * math.pi / 2),
        (5.0, 10.0, 3 * math.pi / 2), (5.0, 40.0, 3 * math.pi / 2),
    ]
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diffraction", description=__doc__)
    parser.add_argument("--results", default="results/run")
    parser.add_argument("--c", type=float, default=SPEED_OF_SOUND, help="speed of sound (m/s)")
    sub = parser.add_subparsers(dest="command", required=True)

    spectrum = sub.add_parser("spectrum", help="one geometry: band insertion loss and the impulse response")
    spectrum.add_argument("--height", type=float, default=3.0)
    spectrum.add_argument("--source-distance", type=float, default=5.0)
    spectrum.add_argument("--receiver-distance", type=float, default=10.0)
    spectrum.add_argument("--source-height", type=float, default=1.5)
    spectrum.add_argument("--receiver-height", type=float, default=1.5)
    spectrum.add_argument("--wedge-deg", type=float, default=360.0)

    study = sub.add_parser("study", help="the full 7 x 6 x 3 parameter study")
    study.add_argument("--centre", type=float, default=500.0, help="band for the marginal means")

    validate = sub.add_parser("validate", help="compare predictions with listening-test ratings")
    validate.add_argument("--ratings", default="../data/listening_test.csv")
    validate.add_argument("--resamples", type=int, default=5000)

    simulate = sub.add_parser("simulate-ratings", help="write a stand-in ratings CSV (clearly not real data)")
    simulate.add_argument("--out", default="../data/listening_test.csv")
    simulate.add_argument("--participants", type=int, default=10)
    simulate.add_argument("--seed", type=int, default=0)
    return parser


def _stimulus_predictions(c: float = SPEED_OF_SOUND) -> Dict[str, float]:
    return {configuration.label: evaluate(configuration, c=c).a_weighted_db for configuration in STIMULI}


def cmd_spectrum(args) -> int:
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)

    wedge = Wedge(open_angle=math.radians(args.wedge_deg), height=args.height)
    source = Point(-args.source_distance, 0.0, args.source_height)
    receiver = Point(args.receiver_distance, 0.0, args.receiver_height)

    def with_barrier(frequencies):
        return barrier_transfer_function(source, receiver, wedge, frequencies, c=args.c)

    def without(frequencies):
        return free_field_transfer_function(source, receiver, frequencies, c=args.c)

    bands = insertion_loss_spectrum(with_barrier, without)
    reference = [maekawa_insertion_loss(fresnel_number(source, receiver, wedge, b.centre, c=args.c)) for b in bands]
    payload = {
        "geometry": {
            "height": args.height,
            "source_distance": args.source_distance,
            "receiver_distance": args.receiver_distance,
            "wedge_deg": args.wedge_deg,
        },
        "path_difference_m": path_difference(source, receiver, wedge),
        "a_weighted_db": a_weighted_insertion_loss(bands),
        "bands": [band.to_dict() for band in bands],
        "maekawa_db": reference,
    }

    samples = impulse_response(with_barrier)
    np.save(results / "impulse_response.npy", samples)
    (results / "spectrum.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (results / "spectrum.md").write_text(
        spectrum_markdown(payload["bands"], maekawa=reference, title="Insertion loss, single geometry"),
        encoding="utf-8",
    )
    band_spectrum(payload["bands"], results / "spectrum.png", reference=reference, title="Insertion loss")
    impulse_response_plot(samples, SAMPLE_RATE, results / "impulse_response.png")

    print(spectrum_markdown(payload["bands"], maekawa=reference, title="Insertion loss, single geometry"))
    print(f"A-weighted insertion loss {payload['a_weighted_db']:.2f} dB; path difference "
          f"{payload['path_difference_m']:.3f} m")
    return 0


def cmd_study(args) -> int:
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    configurations = grid()
    print(f"evaluating {len(configurations)} configurations")
    results = run(configurations, c=args.c)
    marginals = sensitivity(results, centre=args.centre)

    (results_dir / "study.csv").write_text(study_csv(results), encoding="utf-8")
    (results_dir / "study.md").write_text(study_markdown(results, marginals, centre=args.centre), encoding="utf-8")
    (results_dir / "study.json").write_text(
        json.dumps({"sensitivity": marginals, "results": [r.to_dict() for r in results]}, indent=2), encoding="utf-8"
    )
    parameter_surface(results, results_dir / "surface_thin_screen.png", centre=args.centre)
    sensitivity_bars(marginals, results_dir / "sensitivity.png")

    print(study_markdown(results, marginals, centre=args.centre))
    return 0


def cmd_validate(args) -> int:
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    path = Path(args.ratings)
    if not path.exists():
        raise SystemExit(f"no ratings at {path}; run `simulate-ratings` or supply your own CSV")

    ratings = load_ratings(path)
    predictions = _stimulus_predictions(args.c)
    outcome = validate_ratings(ratings, predictions, resamples=args.resamples)

    # Whether the ratings are real is recorded next to them, not guessed from the
    # numbers. An unknown provenance is reported as simulated, because a file with
    # no stated origin is not evidence.
    provenance = read_provenance(path)
    markdown = validation_markdown(outcome, simulated=provenance.get("source") != "listening_test")
    (results_dir / "validation.md").write_text(markdown, encoding="utf-8")
    (results_dir / "validation.json").write_text(json.dumps(outcome.to_dict(), indent=2), encoding="utf-8")
    listening_scatter(
        [predictions[s] for s in ratings.stimuli],
        normalise_within_participant(ratings),
        ratings.stimuli,
        results_dir / "listening.png",
        rho=outcome.spearman_normalised,
    )
    print(markdown)
    return 0


def cmd_simulate(args) -> int:
    predictions = _stimulus_predictions(args.c)
    ratings = simulate_ratings(predictions, n_participants=args.participants, seed=args.seed)
    path = write_ratings(
        ratings,
        args.out,
        provenance={
            "source": "simulated",
            "generator": "diffraction.listening.simulate_ratings",
            "seed": args.seed,
            "note": "Not collected from listeners. Replace with a real study before citing any of it.",
        },
    )
    print(f"wrote {path}: {ratings.n_participants} simulated participants x {ratings.n_stimuli} stimuli")
    print("These are NOT real listener ratings. See the README.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "spectrum":
        return cmd_spectrum(args)
    if args.command == "study":
        return cmd_study(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "simulate-ratings":
        return cmd_simulate(args)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
