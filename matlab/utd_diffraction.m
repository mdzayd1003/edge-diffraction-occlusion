function D = utd_diffraction(g, k)
% UTD_DIFFRACTION  Kouyoumjian & Pathak diffraction coefficient for a rigid wedge.
%
%   D = UTD_DIFFRACTION(G, K) with G from WEDGE_GEOMETRY and K a vector of
%   wavenumbers. The four terms are the two shadow boundaries and the two
%   reflection boundaries; both faces are rigid, so all four enter with a plus.
%
%   At a boundary the cotangent diverges while F vanishes. Their product is
%   finite and its limit is used directly rather than multiplying infinity by
%   zero and hoping.

    k = k(:).';
    L = g.s * g.sPrime * sin(g.beta0)^2 / (g.s + g.sPrime);
    kL = k * L;

    total = zeros(size(k));
    for beta = [g.thetaR - g.thetaS, g.thetaR + g.thetaS]
        for sgn = [1, -1]
            total = total + cot_times_F(beta, sgn, g.n, kL);
        end
    end

    D = -exp(-1j*pi/4) ./ (2 * g.n * sqrt(2*pi*k) * sin(g.beta0)) .* total;
end

function out = cot_times_F(beta, sgn, n, kL)
    N     = round((beta + sgn*pi) / (2*pi*n));
    a     = 2 * cos((2*pi*n*N - beta) / 2)^2;
    arg   = (pi + sgn*beta) / (2*n);
    sinArg = sin(arg);

    if abs(sinArg) > 1e-6
        out = (cos(arg) / sinArg) * utd_transition(kL * a);
    else
        epsilon = beta - (2*pi*n*N - sgn*pi);
        direction = 1; if epsilon < 0, direction = -1; end
        out = n * (sqrt(2*pi*kL) * direction - 2*kL*epsilon*exp(1j*pi/4)) * exp(1j*pi/4);
    end
end
