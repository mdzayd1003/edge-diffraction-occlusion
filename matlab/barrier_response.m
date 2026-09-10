function [H, Hfree] = barrier_response(source, receiver, wedgeAngle, height, f, c)
% BARRIER_RESPONSE  Transfer functions with and without the barrier.
%
%   [H, HFREE] = BARRIER_RESPONSE(SOURCE, RECEIVER, WEDGEANGLE, HEIGHT, F, C)
%   returns the complex pressure at the receiver per unit source strength, with
%   the barrier present (diffracted, plus direct sound when the receiver can see
%   the source) and in free field.

    if nargin < 6 || isempty(c), c = 343; end
    f = f(:).';
    if any(f <= 0)
        error('barrier_response:frequency', 'frequencies must be positive');
    end

    g = wedge_geometry(source, receiver, wedgeAngle, height);
    k = 2*pi*f / c;

    D = utd_diffraction(g, k);
    H = D .* exp(-1j*k*(g.s + g.sPrime)) ./ sqrt(g.s * g.sPrime * (g.s + g.sPrime));

    if ~g.shadowed
        H = H + exp(-1j*k*g.directDistance) / g.directDistance;
    end

    Hfree = exp(-1j*k*g.directDistance) / g.directDistance;
end
