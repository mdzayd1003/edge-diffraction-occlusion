function [IL, centres] = third_octave_insertion_loss(source, receiver, wedgeAngle, height, c)
% THIRD_OCTAVE_INSERTION_LOSS  Band insertion loss, 50 Hz to 10 kHz.
%
%   Energy is integrated across each band on a log-frequency axis rather than
%   sampled at the centre: diffraction is smooth in frequency, but not near a
%   shadow boundary, and integrating costs nothing.

    if nargin < 5 || isempty(c), c = 343; end

    centres = [50 63 80 100 125 160 200 250 315 400 500 630 800 ...
               1000 1250 1600 2000 2500 3150 4000 5000 6300 8000 10000];
    ratio = 2^(1/6);
    IL = zeros(size(centres));

    for i = 1:numel(centres)
        f = logspace(log10(centres(i)/ratio), log10(centres(i)*ratio), 9);
        [H, Hfree] = barrier_response(source, receiver, wedgeAngle, height, f, c);
        logf = log(f);
        withBarrier = trapz(logf, abs(H).^2)    / (logf(end) - logf(1));
        without     = trapz(logf, abs(Hfree).^2) / (logf(end) - logf(1));
        IL(i) = 10*log10(without / max(withBarrier, 1e-30));
    end
end
