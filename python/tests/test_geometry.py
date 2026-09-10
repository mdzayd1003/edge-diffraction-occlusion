import math

import numpy as np
import pytest

from diffraction.geometry import (
    Point,
    Wedge,
    apex_position,
    direct_distance,
    fresnel_number,
    is_shadowed,
    maekawa_insertion_loss,
    path_difference,
    shortest_diffracted_path,
    to_edge_coordinates,
)


def test_a_thin_screen_has_a_wedge_index_of_a_half():
    assert Wedge(open_angle=2 * math.pi).index == pytest.approx(0.5)


def test_an_impossible_open_angle_is_rejected():
    for angle in (0.0, -1.0, 7.0):
        with pytest.raises(ValueError):
            Wedge(open_angle=angle)


@pytest.mark.parametrize("angle", [2 * math.pi, 7 * math.pi / 4, 3 * math.pi / 2])
def test_angles_land_inside_the_open_region_for_any_wedge(angle, source, receiver):
    wedge = Wedge(open_angle=angle, height=3.0)
    for point in (source, receiver):
        theta = to_edge_coordinates(point, wedge).theta
        assert 0 <= theta <= angle, math.degrees(theta)


def test_the_faces_tilt_up_as_the_wedge_fattens(source):
    thin = to_edge_coordinates(source, Wedge(open_angle=2 * math.pi, height=3.0)).theta
    fat = to_edge_coordinates(source, Wedge(open_angle=3 * math.pi / 2, height=3.0)).theta
    assert fat == pytest.approx(thin - math.pi / 4)


def test_the_perpendicular_distance_does_not_depend_on_the_wedge_angle(source):
    distances = {
        round(to_edge_coordinates(source, Wedge(open_angle=a, height=3.0)).r, 9)
        for a in (2 * math.pi, 3 * math.pi / 2)
    }
    assert len(distances) == 1


def test_the_shadow_test_is_symmetric(thin_screen, source, receiver):
    s = to_edge_coordinates(source, thin_screen)
    r = to_edge_coordinates(receiver, thin_screen)
    assert is_shadowed(s, r) == is_shadowed(r, s)
    assert is_shadowed(s, r)


def test_a_receiver_above_the_barrier_is_not_shadowed(thin_screen, source):
    high = to_edge_coordinates(Point(10.0, 0.0, 20.0), thin_screen)
    assert not is_shadowed(to_edge_coordinates(source, thin_screen), high)


def test_the_boundary_sits_at_exactly_pi_of_separation(thin_screen, source, shadow_boundary_receiver):
    s = to_edge_coordinates(source, thin_screen)
    r = to_edge_coordinates(shadow_boundary_receiver, thin_screen)
    assert abs(r.theta - s.theta) == pytest.approx(math.pi, abs=1e-9)


def test_the_apex_splits_the_along_edge_offset_in_proportion(thin_screen):
    s = to_edge_coordinates(Point(-5.0, 0.0, 1.5), thin_screen)
    r = to_edge_coordinates(Point(10.0, 30.0, 1.5), thin_screen)
    apex = apex_position(s, r)
    assert 0 < apex < 30
    assert apex == pytest.approx(30.0 * s.r / (s.r + r.r))


def test_the_diffracted_path_is_never_shorter_than_the_direct_one(thin_screen, source, receiver):
    s = to_edge_coordinates(source, thin_screen)
    r = to_edge_coordinates(receiver, thin_screen)
    assert shortest_diffracted_path(s, r) >= direct_distance(source, receiver)


def test_the_path_difference_vanishes_at_the_shadow_boundary(thin_screen, source, shadow_boundary_receiver):
    assert path_difference(source, shadow_boundary_receiver, thin_screen) == pytest.approx(0.0, abs=1e-9)


def test_a_taller_barrier_lengthens_the_detour(source, receiver):
    deltas = [path_difference(source, receiver, Wedge(open_angle=2 * math.pi, height=h)) for h in (2, 3, 4, 6)]
    assert all(b > a for a, b in zip(deltas, deltas[1:]))


def test_the_fresnel_number_is_positive_in_shadow_and_negative_in_light(thin_screen, source, receiver):
    assert fresnel_number(source, receiver, thin_screen, 500.0) > 0
    assert fresnel_number(source, Point(10.0, 0.0, 20.0), thin_screen, 500.0) < 0


def test_the_fresnel_number_scales_with_frequency(thin_screen, source, receiver):
    low = fresnel_number(source, receiver, thin_screen, 500.0)
    high = fresnel_number(source, receiver, thin_screen, 1000.0)
    assert high == pytest.approx(2 * low)


def test_a_non_positive_frequency_is_rejected(thin_screen, source, receiver):
    with pytest.raises(ValueError):
        fresnel_number(source, receiver, thin_screen, 0.0)


def test_maekawa_is_five_db_at_the_boundary_and_grows_with_the_fresnel_number():
    assert maekawa_insertion_loss(0.0) == pytest.approx(5.0)
    values = [maekawa_insertion_loss(n) for n in (0.1, 1, 3, 10, 30)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_maekawa_gives_nothing_away_deep_in_the_bright_zone():
    assert maekawa_insertion_loss(-5.0) == 0.0
    assert 0.0 <= maekawa_insertion_loss(-0.2) <= 5.0
