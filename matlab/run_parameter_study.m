function results = run_parameter_study(outputDir)
% RUN_PARAMETER_STUDY  7 barrier heights x 6 receiver distances x 3 wedge angles.
%
%   RESULTS = RUN_PARAMETER_STUDY(OUTPUTDIR) evaluates all 126 configurations and,
%   when OUTPUTDIR is given, writes study.csv there in the same layout as the
%   Python port, so the two can be diffed against each other.

    if nargin < 1, outputDir = ''; end

    heights   = [2.0 2.5 3.0 3.5 4.0 5.0 6.0];
    distances = [5 10 20 40 60 80];
    angles    = [2*pi, 7*pi/4, 3*pi/2];

    sourceDistance = 5.0; sourceHeight = 1.5; receiverHeight = 1.5;

    results = struct('height', {}, 'distance', {}, 'wedgeDeg', {}, ...
                     'pathDifference', {}, 'centres', {}, 'IL', {});

    for h = heights
        for d = distances
            for a = angles
                source   = [-sourceDistance, 0, sourceHeight];
                receiver = [ d,              0, receiverHeight];
                [IL, centres] = third_octave_insertion_loss(source, receiver, a, h);
                g = wedge_geometry(source, receiver, a, h);

                results(end+1) = struct( ...
                    'height', h, 'distance', d, 'wedgeDeg', a*180/pi, ...
                    'pathDifference', g.pathDifference, 'centres', centres, 'IL', IL); %#ok<AGROW>
            end
        end
    end

    fprintf('evaluated %d configurations\n', numel(results));

    if ~isempty(outputDir)
        if ~exist(outputDir, 'dir'), mkdir(outputDir); end
        path = fullfile(outputDir, 'study.csv');
        fid = fopen(path, 'w');
        fprintf(fid, 'height_m,receiver_distance_m,wedge_angle_deg,path_difference_m');
        fprintf(fid, ',il_%ghz_db', results(1).centres);
        fprintf(fid, '\n');
        for i = 1:numel(results)
            fprintf(fid, '%g,%g,%.0f,%.4f', results(i).height, results(i).distance, ...
                    results(i).wedgeDeg, results(i).pathDifference);
            fprintf(fid, ',%.3f', results(i).IL);
            fprintf(fid, '\n');
        end
        fclose(fid);
        fprintf('wrote %s\n', path);
    end
end
