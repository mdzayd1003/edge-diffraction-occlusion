function out = validate_listening_test(csvPath, predictions, stimuli, resamples)
% VALIDATE_LISTENING_TEST  Compare predicted insertion loss with listener ratings.
%
%   Ratings are z-scored within participant before averaging: people use a 1-7
%   scale differently, and that is individual scale use rather than disagreement
%   about the stimuli. The correlation is Spearman, because the prediction is in
%   dB and the rating is ordinal — only the ordering is comparable.
%
%   The bootstrap resamples PARTICIPANTS, not individual ratings. Ten people is
%   the sample size; treating 120 ratings as independent would give an interval
%   several times too narrow.

    if nargin < 4 || isempty(resamples), resamples = 5000; end

    [participants, ~, matrix] = read_ratings(csvPath, stimuli);

    normalised = zeros(size(matrix));
    for i = 1:size(matrix, 1)
        row = matrix(i, :);
        sd = std(row);
        if sd < 1e-12, sd = 1; end
        normalised(i, :) = (row - mean(row)) / sd;
    end

    out.nParticipants = numel(participants);
    out.nStimuli = numel(stimuli);
    out.spearmanNormalised = spearman(mean(normalised, 1), predictions);
    out.spearmanRaw = spearman(mean(matrix, 1), predictions);

    values = zeros(1, resamples);
    for b = 1:resamples
        index = randi(out.nParticipants, 1, out.nParticipants);
        values(b) = spearman(mean(normalised(index, :), 1), predictions);
    end
    values = sort(values);
    out.ciLow  = values(max(1, floor(0.025 * resamples)));
    out.ciHigh = values(min(resamples, ceil(0.975 * resamples)));

    fprintf('Spearman rho = %.3f, 95%% CI [%.3f, %.3f] over %d participants\n', ...
            out.spearmanNormalised, out.ciLow, out.ciHigh, out.nParticipants);
end

function [participants, stimuli, matrix] = read_ratings(csvPath, stimuli)
    fid = fopen(csvPath, 'r');
    if fid < 0, error('validate_listening_test:open', 'cannot read %s', csvPath); end
    header = fgetl(fid); %#ok<NASGU>
    rows = textscan(fid, '%s%s%f', 'Delimiter', ',');
    fclose(fid);

    participants = unique(rows{1});
    matrix = nan(numel(participants), numel(stimuli));
    for i = 1:numel(rows{3})
        p = find(strcmp(participants, rows{1}{i}));
        s = find(strcmp(stimuli, rows{2}{i}));
        if ~isempty(p) && ~isempty(s)
            matrix(p, s) = rows{3}(i);
        end
    end
    if any(isnan(matrix(:)))
        error('validate_listening_test:incomplete', 'the design has missing participant/stimulus cells');
    end
end

function rho = spearman(x, y)
    rho = pearson(average_rank(x), average_rank(y));
end

function r = average_rank(x)
    [sorted, order] = sort(x(:).');
    r = zeros(1, numel(x));
    r(order) = 1:numel(x);
    start = 1;
    for i = 2:numel(sorted)+1
        if i == numel(sorted)+1 || sorted(i) ~= sorted(start)
            if i - start > 1
                r(order(start:i-1)) = mean(r(order(start:i-1)));
            end
            start = i;
        end
    end
end

function c = pearson(x, y)
    x = x(:) - mean(x); y = y(:) - mean(y);
    denominator = sqrt((x'*x) * (y'*y));
    if denominator == 0, c = 0; else, c = (x'*y) / denominator; end
end
