function failures = run_tests()
% RUN_TESTS  The MATLAB port's test suite. Runs unmodified under Octave.
%
%   octave --no-gui --eval "addpath('..'); exit(run_tests())"
%
%   Mirrors the Python suite's checks, so a divergence between the two ports
%   shows up as a failing assertion rather than as two different answers nobody
%   compared.

    addpath(fileparts(fileparts(mfilename('fullpath'))));

    tests = {
        @test_transition_function_limits
        @test_geometry_angles_lie_inside_the_open_region
        @test_a_point_inside_the_wedge_body_is_rejected
        @test_shadow_test_is_symmetric
        @test_reciprocity
        @test_half_field_at_the_shadow_boundary
        @test_field_is_continuous_across_the_shadow_boundary
        @test_matches_maekawa_within_a_few_db
        @test_insertion_loss_rises_with_frequency
        @test_a_taller_barrier_attenuates_more
        @test_maekawa_is_zero_deep_in_the_bright_zone
        @test_band_insertion_loss_covers_the_standard_centres
    };

    failures = 0;
    for i = 1:numel(tests)
        name = func2str(tests{i});
        try
            tests{i}();
            fprintf('  ok    %s\n', name);
        catch err
            failures = failures + 1;
            fprintf('  FAIL  %s: %s\n', name, err.message);
        end
    end
    fprintf('%d/%d passed\n', numel(tests) - failures, numel(tests));
end

% -- individual tests ------------------------------------------------------

function test_transition_function_limits()
    assert(abs(utd_transition(0)) < 1e-12, 'F(0) must vanish');
    assert(abs(abs(utd_transition(100)) - 1) < 1e-2, 'F must approach 1 for large X');
    assert(abs(utd_transition(1)) > 0.7 && abs(utd_transition(1)) < 0.95, 'F(1) magnitude');
end

function test_geometry_angles_lie_inside_the_open_region()
    for a = [2*pi, 7*pi/4, 3*pi/2]
        g = wedge_geometry([-5 0 1.5], [10 0 1.5], a, 3);
        assert(g.thetaS >= 0 && g.thetaS <= a, 'source angle inside the open region');
        assert(g.thetaR >= 0 && g.thetaR <= a, 'receiver angle inside the open region');
    end
end

function test_a_point_inside_the_wedge_body_is_rejected()
    threw = false;
    try
        wedge_geometry([-5 0 1.5], [10 0 1.5], pi/2, 3);
    catch
        threw = true;
    end
    assert(threw, 'a receiver inside the wedge body must be refused');
end

function test_shadow_test_is_symmetric()
    a = wedge_geometry([-5 0 1.5], [10 0 1.5], 2*pi, 3);
    b = wedge_geometry([10 0 1.5], [-5 0 1.5], 2*pi, 3);
    assert(a.shadowed == b.shadowed, 'the shadow test must not depend on which end is the source');
    assert(a.shadowed, 'this geometry is shadowed');
end

function test_reciprocity()
    f = [125 500 2000 8000];
    H1 = barrier_response([-5 0 1.5], [10 4 1.8], 2*pi, 3, f);
    H2 = barrier_response([10 4 1.8], [-5 0 1.5], 2*pi, 3, f);
    assert(max(abs(H1 - H2) ./ abs(H1)) < 1e-9, 'swapping source and receiver must change nothing');
end

function test_half_field_at_the_shadow_boundary()
    % A receiver exactly on the line from the source over the edge.
    zb = 1.5 + (3 - 1.5) * (10 + 5) / 5;
    f = [1000 4000];
    [H, Hfree] = barrier_response([-5 0 1.5], [10 0 zb], 2*pi, 3, f);
    ratio = abs(H) ./ abs(Hfree);
    assert(all(abs(ratio - 0.5) < 0.06), ...
           sprintf('at the shadow boundary the field must be about half; got %s', mat2str(ratio, 3)));
end

function test_field_is_continuous_across_the_shadow_boundary()
    zb = 1.5 + (3 - 1.5) * (10 + 5) / 5;
    f = 1000;
    levels = zeros(1, 5);
    offsets = [-0.10 -0.05 0 0.05 0.10];
    for i = 1:numel(offsets)
        [H, Hfree] = barrier_response([-5 0 1.5], [10 0 zb + offsets(i)], 2*pi, 3, f);
        levels(i) = abs(H) / abs(Hfree);
    end
    assert(all(abs(diff(levels)) < 0.05), 'no step is allowed across the boundary');
    assert(all(diff(levels) > 0), 'the field must grow as the receiver leaves the shadow');
end

function test_matches_maekawa_within_a_few_db()
    source = [-5 0 1.5]; receiver = [10 0 1.5]; height = 3; c = 343;
    g = wedge_geometry(source, receiver, 2*pi, height);
    f = [125 500 2000 8000];
    [H, Hfree] = barrier_response(source, receiver, 2*pi, height, f, c);
    IL = 20*log10(abs(Hfree) ./ abs(H));
    N = 2 * g.pathDifference * f / c;
    reference = maekawa_insertion_loss(N);
    assert(all(abs(IL - reference) < 3), ...
           sprintf('UTD should sit within 3 dB of Maekawa; got %s', mat2str(IL - reference, 3)));
end

function test_insertion_loss_rises_with_frequency()
    f = [125 250 500 1000 2000 4000];
    [H, Hfree] = barrier_response([-5 0 1.5], [10 0 1.5], 2*pi, 3, f);
    IL = 20*log10(abs(Hfree) ./ abs(H));
    assert(all(diff(IL) > 0), 'a barrier attenuates high frequencies more');
end

function test_a_taller_barrier_attenuates_more()
    f = 500;
    previous = -inf;
    for h = [2 3 4 6]
        [H, Hfree] = barrier_response([-5 0 1.5], [10 0 1.5], 2*pi, h, f);
        IL = 20*log10(abs(Hfree) / abs(H));
        assert(IL > previous, 'insertion loss must grow with barrier height');
        previous = IL;
    end
end

function test_maekawa_is_zero_deep_in_the_bright_zone()
    assert(maekawa_insertion_loss(-5) == 0, 'no attenuation well into the bright zone');
    assert(abs(maekawa_insertion_loss(0) - 5) < 1e-9, '5 dB at the boundary');
    assert(maekawa_insertion_loss(10) > maekawa_insertion_loss(1), 'monotone in the Fresnel number');
end

function test_band_insertion_loss_covers_the_standard_centres()
    [IL, centres] = third_octave_insertion_loss([-5 0 1.5], [10 0 1.5], 2*pi, 3);
    assert(numel(centres) == 24, '50 Hz to 10 kHz is 24 third-octave bands');
    assert(numel(IL) == numel(centres), 'one value per band');
    assert(all(isfinite(IL)), 'every band must be finite');
    assert(IL(end) > IL(1), 'high bands are attenuated more than low ones');
end
