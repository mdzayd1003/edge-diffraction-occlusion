function g = wedge_geometry(source, receiver, wedgeAngle, height)
% WEDGE_GEOMETRY  Express a source and receiver in a wedge edge's frame.
%
%   g = WEDGE_GEOMETRY(SOURCE, RECEIVER, WEDGEANGLE, HEIGHT) takes two [x y z]
%   world points, the wedge's open angle in radians and the height of its edge,
%   and returns a struct with the quantities the diffraction integral needs.
%
%   The barrier's top edge runs along y at z = HEIGHT in the plane x = 0. Angles
%   are measured from the source-side face and increase through the open region,
%   so theta lies in [0, WEDGEANGLE] for any point outside the wedge body. The
%   solid part is centred on the downward vertical, which is why each face is
%   tilted up by (2*pi - WEDGEANGLE)/2.
%
%   Fields: rS, thetaS, zS, rR, thetaR, zR, n (= WEDGEANGLE/pi), apex,
%   sPrime, s, beta0, shadowed, directDistance, pathDifference.
%
%   Octave-compatible: no toolbox functions.

    if wedgeAngle <= 0 || wedgeAngle > 2*pi
        error('wedge_geometry:angle', 'open angle must lie in (0, 2*pi]');
    end

    halfSolid = (2*pi - wedgeAngle) / 2;

    [g.rS, g.thetaS, g.zS] = local_coords(source, height, halfSolid);
    [g.rR, g.thetaR, g.zR] = local_coords(receiver, height, halfSolid);

    if g.rS <= 0 || g.rR <= 0
        error('wedge_geometry:onEdge', 'source and receiver must be off the edge itself');
    end
    if g.thetaS > wedgeAngle + 1e-9 || g.thetaR > wedgeAngle + 1e-9
        error('wedge_geometry:insideBody', ...
              'a point lies inside the body of the wedge (theta = %.1f deg, max %.1f deg)', ...
              max(g.thetaS, g.thetaR)*180/pi, wedgeAngle*180/pi);
    end

    g.n = wedgeAngle / pi;

    % Keller's cone: the diffraction point splits the along-edge offset in
    % proportion to the perpendicular distances. Symmetric in source/receiver,
    % which is what keeps the whole thing reciprocal.
    g.apex   = g.zS + (g.zR - g.zS) * g.rS / (g.rS + g.rR);
    g.sPrime = hypot(g.rS, g.apex - g.zS);
    g.s      = hypot(g.rR, g.apex - g.zR);
    g.beta0  = asin(min(1, g.rS / g.sPrime));

    % Unsigned angular separation, so swapping source and receiver cannot change
    % the answer. A difference taken modulo 2*pi is not symmetric and quietly
    % breaks reciprocity.
    g.shadowed = abs(g.thetaR - g.thetaS) > pi;

    g.directDistance = norm(source(:) - receiver(:));
    g.pathDifference = g.sPrime + g.s - g.directDistance;
end

function [r, theta, z] = local_coords(point, height, halfSolid)
    dx = point(1);
    dz = point(3) - height;
    r  = hypot(dx, dz);
    theta = mod(atan2(-dx, -dz) - halfSolid, 2*pi);
    z = point(2);
end
