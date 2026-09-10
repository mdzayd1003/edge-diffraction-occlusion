import math

import numpy as np
import pytest

from diffraction.geometry import Point, Wedge, fresnel_number, maekawa_insertion_loss
from diffraction.utd import (
    barrier_transfer_function,
    diffraction_coefficient,
    free_field_transfer_function,
    transition_function,
)


def insertion_loss(source, receiver, wedge, frequencies):
    barrier = barrier_transfer_function(source, receiver, wedge, frequencies)
    free = free_field_transfer_function(source, receiver, frequencies)
    return 20 * np.log10(np.abs(free) / np.abs(barrier))


def test_the_transition_function_vanishes_at_zero_and_saturates():
    assert abs(transition_function(np.array([0.0]))[0]) == pytest.approx(0.0, abs=1e-12)
    assert abs(transition_function(np.array([100.0]))[0]) == pytest.approx(1.0, abs=1e-3)
    assert abs(transition_function(np.array([1000.0]))[0]) == pytest.approx(1.0, abs=1e-3)


def test_the_transition_function_is_monotone_in_magnitude():
    values = np.abs(transition_function(np.array([0.01, 0.1, 1.0, 10.0, 100.0])))
    assert all(b > a for a, b in zip(values, values[1:]))


def test_the_transition_phase_tends_to_forty_five_degrees_at_the_boundary():
    phase = math.degrees(np.angle(transition_function(np.array([1e-4]))[0]))
    assert phase == pytest.approx(45.0, abs=1.0)


def test_the_diffraction_coefficient_falls_like_one_over_root_k():
    # Well above the transition region (kL ≫ 1) F has saturated at 1, so only the
    # 1/√k prefactor is left moving: quadrupling k must halve the coefficient.
    k = np.array([1e4, 4e4])
    values = np.abs(diffraction_coefficient(1.0, 4.0, 2.0, k, np.array([3.0, 3.0])))
    assert values[0] / values[1] == pytest.approx(2.0, rel=0.02)


def test_the_field_is_exactly_half_at_the_shadow_boundary(thin_screen, source, shadow_boundary_receiver):
    """Nothing is fitted to produce this; it falls out of the transition function."""
    frequencies = np.array([1000.0, 4000.0, 16000.0])
    barrier = barrier_transfer_function(source, shadow_boundary_receiver, thin_screen, frequencies)
    free = free_field_transfer_function(source, shadow_boundary_receiver, frequencies)
    ratio = np.abs(barrier) / np.abs(free)
    assert np.allclose(ratio, 0.5, atol=0.03), ratio


def test_the_field_is_continuous_across_the_shadow_boundary(thin_screen, source, shadow_boundary_receiver):
    frequencies = np.array([1000.0])
    levels = []
    for offset in (-0.10, -0.05, 0.0, 0.05, 0.10):
        receiver = Point(shadow_boundary_receiver.x, 0.0, shadow_boundary_receiver.z + offset)
        barrier = barrier_transfer_function(source, receiver, thin_screen, frequencies)
        free = free_field_transfer_function(source, receiver, frequencies)
        levels.append(float((np.abs(barrier) / np.abs(free))[0]))
    assert all(abs(b - a) < 0.05 for a, b in zip(levels, levels[1:])), levels
    assert all(b > a for a, b in zip(levels, levels[1:])), "the field grows on leaving the shadow"


@pytest.mark.parametrize(
    "receiver", [Point(10.0, 0.0, 1.5), Point(10.0, 4.0, 1.8), Point(6.0, -2.0, 2.2)]
)
def test_reciprocity_holds_to_machine_precision(thin_screen, source, receiver, frequencies):
    forward = barrier_transfer_function(source, receiver, thin_screen, frequencies)
    backward = barrier_transfer_function(receiver, source, thin_screen, frequencies)
    np.testing.assert_allclose(forward, backward, rtol=1e-10)


def test_it_agrees_with_maekawa_within_two_decibels(thin_screen, source, receiver):
    """Maekawa is empirical, so this is a check against measurement, not theory."""
    frequencies = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    predicted = insertion_loss(source, receiver, thin_screen, frequencies)
    reference = np.array(
        [maekawa_insertion_loss(fresnel_number(source, receiver, thin_screen, f)) for f in frequencies]
    )
    difference = predicted - reference
    assert np.all(np.abs(difference) < 2.0), difference
    assert np.all(difference > 0), "UTD is known to sit slightly above the Maekawa chart"


def test_insertion_loss_grows_with_frequency(thin_screen, source, receiver):
    values = insertion_loss(source, receiver, thin_screen, np.array([125.0, 250.0, 500.0, 1000.0, 2000.0]))
    assert all(b > a for a, b in zip(values, values[1:]))


def test_a_taller_barrier_attenuates_more(source, receiver):
    values = [
        float(insertion_loss(source, receiver, Wedge(open_angle=2 * math.pi, height=h), np.array([500.0]))[0])
        for h in (2, 3, 4, 6)
    ]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_a_fatter_wedge_diffracts_more_energy_into_the_shadow(source, receiver):
    """Single-edge behaviour: smaller open angle, larger coefficient, less loss."""
    values = [
        float(insertion_loss(source, receiver, Wedge(open_angle=a, height=3.0), np.array([500.0]))[0])
        for a in (3 * math.pi / 2, 7 * math.pi / 4, 2 * math.pi)
    ]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_a_receiver_well_above_the_barrier_hears_almost_the_free_field(thin_screen, source):
    loss = insertion_loss(source, Point(10.0, 0.0, 30.0), thin_screen, np.array([2000.0]))
    assert abs(float(loss[0])) < 1.0


def test_a_point_inside_the_wedge_body_is_refused(source, receiver):
    with pytest.raises(ValueError, match="inside the body"):
        barrier_transfer_function(source, receiver, Wedge(open_angle=math.pi / 2, height=3.0), np.array([500.0]))


def test_a_point_on_the_edge_itself_is_refused(thin_screen, receiver):
    with pytest.raises(ValueError, match="off the edge"):
        barrier_transfer_function(Point(0.0, 0.0, 3.0), receiver, thin_screen, np.array([500.0]))


def test_dc_is_refused_rather_than_returned_as_infinity(thin_screen, source, receiver):
    with pytest.raises(ValueError, match="DC"):
        barrier_transfer_function(source, receiver, thin_screen, np.array([0.0]))


def test_the_free_field_is_a_plain_spherical_wave(source, receiver):
    frequencies = np.array([500.0])
    value = free_field_transfer_function(source, receiver, frequencies)[0]
    distance = math.dist(source.as_array(), receiver.as_array())
    assert abs(value) == pytest.approx(1.0 / distance)
