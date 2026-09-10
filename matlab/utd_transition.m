function F = utd_transition(X)
% UTD_TRANSITION  The UTD transition function F(X).
%
%   F(X) = 2j*sqrt(X)*exp(1j*X) * integral from sqrt(X) to inf of exp(-1j*t^2) dt
%
%   F -> 0 as X -> 0 (at a shadow or reflection boundary, where the cotangent it
%   multiplies is diverging) and F -> 1 for X >> 1 (geometrical optics). The tail
%   integral is written in terms of Fresnel integrals, which are computed here
%   directly so no toolbox is needed.

    X = max(X, 0);
    a = sqrt(2 * X / pi);
    [S, C] = fresnel_integrals(a);
    tail = sqrt(pi/2) * ((0.5 - C) - 1j * (0.5 - S));
    F = 2j .* sqrt(X) .* exp(1j .* X) .* tail;
end

function [S, C] = fresnel_integrals(x)
% Fresnel S(x) = int_0^x sin(pi t^2/2) dt, C(x) = int_0^x cos(pi t^2/2) dt.
% Adaptive Simpson on a fixed fine grid: the integrand oscillates faster as x
% grows, so the number of panels grows with x rather than being constant.
    S = zeros(size(x));
    C = zeros(size(x));
    for k = 1:numel(x)
        upper = x(k);
        if upper <= 0
            continue;
        end
        panels = max(200, ceil(200 * upper));
        panels = panels + mod(panels, 2);           % Simpson needs an even count
        t = linspace(0, upper, panels + 1);
        h = t(2) - t(1);
        w = ones(1, panels + 1);
        w(2:2:end-1) = 4;
        w(3:2:end-2) = 2;
        S(k) = h/3 * sum(w .* sin(pi * t.^2 / 2));
        C(k) = h/3 * sum(w .* cos(pi * t.^2 / 2));
    end
end
