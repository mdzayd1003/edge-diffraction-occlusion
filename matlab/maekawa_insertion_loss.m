function IL = maekawa_insertion_loss(N)
% MAEKAWA_INSERTION_LOSS  Maekawa's empirical barrier chart in closed form.
%
%   An independent reference: it comes from measurements, not from the
%   diffraction theory, so agreement between the two means something. Used to
%   check the UTD implementation, never as a substitute for it.

    IL = zeros(size(N));
    for i = 1:numel(N)
        if N(i) > 0
            a = sqrt(2*pi*N(i));
            IL(i) = 5 + 20*log10(a / tanh(a));
        elseif N(i) == 0
            IL(i) = 5;
        else
            a = sqrt(2*pi*abs(N(i)));
            if a >= pi/2
                IL(i) = 0;
            else
                IL(i) = max(0, 5 - 20*log10(tan(a) / a));
            end
        end
    end
end
