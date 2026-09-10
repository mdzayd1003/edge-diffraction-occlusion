import math

import numpy as np
import pytest

from diffraction import SAMPLE_RATE
from diffraction.bands import (
    THIRD_OCTAVE_CENTRES,
    a_weighted_insertion_loss,
    a_weighting_db,
    band_edges,
    band_energy,
    band_frequencies,
    impulse_response,
    insertion_loss_spectrum,
)
from diffraction.geometry import Point, Wedge
from diffraction.study import Configuration, evaluate, grid, sensitivity
from diffraction.utd import barrier_transfer_function, free_field_transfer_function


def test_the_bands_are_third_octaves_that_tile_the_range():
    assert len(THIRD_OCTAVE_CENTRES) == 24
    for low, high in zip(THIRD_OCTAVE_CENTRES, THIRD_OCTAVE_CENTRES[1:]):
        assert high / low == pytest.approx(2 ** (1 / 3), rel=0.03)


def test_band_edges_straddle_the_centre():
    low, high = band_edges(1000.0)
    assert low < 1000 < high
    assert high / low == pytest.approx(2 ** (1 / 3), rel=1e-9)


def test_a_non_positive_centre_is_rejected():
    with pytest.raises(ValueError):
        band_edges(0.0)


def test_band_sampling_spans_the_band():
    frequencies = band_frequencies(1000.0, 9)
    assert len(frequencies) == 9
    assert frequencies[0] == pytest.approx(band_edges(1000.0)[0])
    assert frequencies[-1] == pytest.approx(band_edges(1000.0)[1])


def test_a_band_needs_at_least_two_points():
    with pytest.raises(ValueError):
        band_frequencies(1000.0, 1)


def test_band_energy_of_a_flat_response_is_its_squared_magnitude():
    frequencies = band_frequencies(1000.0)
    assert band_energy(frequencies, np.full_like(frequencies, 2.0)) == pytest.approx(4.0)


def test_a_weighting_is_zero_at_a_kilohertz_and_negative_at_the_extremes():
    assert a_weighting_db(np.array([1000.0]))[0] == pytest.approx(0.0, abs=0.2)
    assert a_weighting_db(np.array([50.0]))[0] < -25
    assert a_weighting_db(np.array([10000.0]))[0] < 0


def test_the_spectrum_has_one_entry_per_band(thin_screen, source, receiver):
    bands = insertion_loss_spectrum(
        lambda f: barrier_transfer_function(source, receiver, thin_screen, f),
        lambda f: free_field_transfer_function(source, receiver, f),
    )
    assert [b.centre for b in bands] == list(map(float, THIRD_OCTAVE_CENTRES))
    assert all(b.insertion_loss_db > 0 for b in bands)


def test_the_a_weighted_figure_sits_inside_the_band_range(thin_screen, source, receiver):
    bands = insertion_loss_spectrum(
        lambda f: barrier_transfer_function(source, receiver, thin_screen, f),
        lambda f: free_field_transfer_function(source, receiver, f),
    )
    values = [b.insertion_loss_db for b in bands]
    assert min(values) < a_weighted_insertion_loss(bands) < max(values)


def test_the_impulse_response_peaks_at_the_diffracted_arrival(thin_screen, source, receiver):
    from diffraction.geometry import shortest_diffracted_path, to_edge_coordinates

    samples = impulse_response(lambda f: barrier_transfer_function(source, receiver, thin_screen, f), n_fft=8192)
    expected = shortest_diffracted_path(
        to_edge_coordinates(source, thin_screen), to_edge_coordinates(receiver, thin_screen)
    ) / 343.0
    peak = int(np.argmax(np.abs(samples))) / SAMPLE_RATE
    assert peak == pytest.approx(expected, abs=2 / SAMPLE_RATE)


def test_the_impulse_response_is_real_and_finite(thin_screen, source, receiver):
    samples = impulse_response(lambda f: barrier_transfer_function(source, receiver, thin_screen, f), n_fft=4096)
    assert samples.dtype == np.float64
    assert len(samples) == 4096
    assert np.isfinite(samples).all()


def test_the_impulse_response_needs_an_even_transform(thin_screen, source, receiver):
    with pytest.raises(ValueError):
        impulse_response(lambda f: barrier_transfer_function(source, receiver, thin_screen, f), n_fft=4097)


def test_the_grid_is_the_full_cross_product():
    configurations = grid()
    assert len(configurations) == 7 * 6 * 3 == 126
    assert len({c.label for c in configurations}) == 126


def test_a_configuration_builds_a_consistent_scene():
    configuration = Configuration(height=3.0, receiver_distance=20.0, wedge_angle=2 * math.pi)
    assert configuration.wedge.height == 3.0
    assert configuration.source.x == -5.0
    assert configuration.receiver.x == 20.0
    assert configuration.label == "h3-d20-w360"


def test_evaluating_one_configuration_gives_bands_and_a_summary():
    result = evaluate(Configuration(height=3.0, receiver_distance=10.0, wedge_angle=2 * math.pi))
    assert len(result.bands) == 24
    assert result.path_difference_m > 0
    assert result.fresnel_500hz > 0
    assert result.band_value(500.0) > 0
    assert result.a_weighted_db > 0


def test_asking_for_a_band_that_was_not_computed_is_an_error():
    result = evaluate(Configuration(height=3.0, receiver_distance=10.0, wedge_angle=2 * math.pi), centres=[500.0])
    with pytest.raises(KeyError):
        result.band_value(1000.0)


@pytest.fixture(scope="module")
def study_results():
    from diffraction.study import run

    return run(grid(), centres=[125.0, 500.0, 2000.0])


def test_insertion_loss_rises_with_barrier_height(study_results):
    marginals = sensitivity(study_results, centre=500.0)["height"]
    values = [marginals[k] for k in sorted(marginals, key=float)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_insertion_loss_falls_slightly_with_receiver_distance(study_results):
    marginals = sensitivity(study_results, centre=500.0)["receiver_distance"]
    values = [marginals[k] for k in sorted(marginals, key=float)]
    assert all(b < a for a, b in zip(values, values[1:]))


def test_a_thin_screen_attenuates_more_than_a_fat_wedge(study_results):
    marginals = sensitivity(study_results, centre=500.0)["wedge_angle"]
    assert marginals["thin screen"] > marginals["315° wedge"] > marginals["270° wedge"]


def test_the_python_port_matches_the_matlab_port(matlab_reference):
    """The two implementations are written independently and must agree.

    A disagreement here means one of them changed and the other did not — which
    is exactly the failure a two-language port is prone to and would otherwise
    be invisible.
    """
    from diffraction.study import run

    results = run(grid())
    assert len(results) == len(matlab_reference)

    worst = 0.0
    for result in results:
        key = (
            result.configuration.height,
            result.configuration.receiver_distance,
            round(math.degrees(result.configuration.wedge_angle)),
        )
        reference = matlab_reference[key]
        ours = [band["insertion_loss_db"] for band in result.bands]
        assert len(ours) == len(reference)
        worst = max(worst, max(abs(a - b) for a, b in zip(ours, reference)))
    assert worst < 0.05, f"largest MATLAB/Python disagreement was {worst:.4f} dB"


def test_the_band_integral_does_not_depend_on_the_numpy_spelling():
    """numpy renamed trapz -> trapezoid in 2.0; both must work.

    Without the binding in bands.py this module is silently numpy-2-only while
    requirements.txt claims >=1.24 — which passes CI (latest numpy) and fails for
    anyone with a pinned environment.
    """
    import numpy as np

    from diffraction.bands import _trapezoid

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 1.0, 1.0])
    assert _trapezoid(y, x) == pytest.approx(2.0)
    assert _trapezoid is getattr(np, "trapezoid", None) or _trapezoid is np.trapz
