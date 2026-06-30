% Define participant pool
participantIDs = 6:13;
blocks = [2, 3];

% Preallocate arrays for the THREE fitted parameters
subj_Hs   = zeros(length(participantIDs), length(blocks));

numBeliefSwitches = zeros(length(participantIDs), length(blocks));
numSourceSwitches = zeros(length(participantIDs), length(blocks));
agentPart = zeros(length(participantIDs), length(blocks));
agentTask = zeros(length(participantIDs), length(blocks));

left = 240;
right = 1680;

% Bounds for [H, w, beta]
lb = [0.000]; %%%%%
ub = [0.999]; %%%%%

% fmincon options (good practice to suppress output so it runs cleaner)
options = optimoptions('fmincon', 'Display', 'off');

for p = 1:length(participantIDs)
    pid = participantIDs(p);
    load(['/Users/janebaker/Desktop/triangles/taskData/real data/p' num2str(pid) '.mat']);

    for b = 1:length(blocks)
        if b == 1
            distStd = 480;
        else
            distStd = 800;
        end

        thisBlockTask = data([data.blockNumber] == blocks(b));

        %% Optimization
        % Initial guesses (true H = 0.003)
        params_init = [0.003]; %%%%%

        objFunc = @(params) objectiveFunction(params, thisBlockTask, left, right, distStd);

        % Run fmincon
        [fitted_params, min_error] = fmincon(objFunc, params_init, ...
            [], [], [], [], lb, ub, [], options);

        % Store fitted parameters
        subj_Hs(p, b)   = fitted_params(1);

        %% count switches
        beliefLeft = [thisBlockTask.keyPressed] == 'F';
        sourceLeft = [thisBlockTask.sourceIsLeft];

        numBeliefSwitches(p, b) = sum(abs(diff(beliefLeft)));
        numSourceSwitches(p, b) = sum(abs(diff(sourceLeft)));

        %% model accuracy
        [tmpBelief, ~] = agentBelief(thisBlockTask, fitted_params(1), ...
            left, right, distStd);  %%%%%
        tmpBelief = tmpBelief.';
        agentLeft = (tmpBelief < 0);
        agentPart(p, b) = sum(beliefLeft == agentLeft) / length(agentLeft);
        agentTask(p,b) = sum(sourceLeft == agentLeft) / length(agentLeft);

    end
end

colors = ['b', 'r'];

fig = figure;
for b = 1:length(blocks)
    h = subj_Hs(:, b);
    s = numBeliefSwitches(:, b);

    plot(h, s, 'o', 'MarkerFaceColor', colors(b), 'MarkerEdgeColor', colors(b))
    hold on
end

xlim([0 1])
ylim([10 40])
xlabel('Subjective Hazard Rate')
ylabel('Number of Belief Switches')

f = figure;
for b = 1:length(blocks)
    h = subj_Hs(:, b);
    s = numSourceSwitches(:, b);

    plot(h, s, 'o', 'MarkerFaceColor', 'auto')
    hold on
end

xlim([0 1])
ylim([10 40])
xlabel('Subjective Hazard Rate')
ylabel('Number of Source Switches')


%{
%% plot normalized LLR with target location, participant belief
blocks = [2 3];

% Set up sampling parameters (e.g., downsampling factor to convert frames to ms)
numFrames = 6000;
dsFactor = 50;
numMs = numFrames * dsFactor;
idxs = 1:numFrames;

% Loop through each participant to extract and process eye-tracking and task data
for p = 1:length(participantIDs)
    pid = participantIDs(p);

    % Add necessary toolboxes and analysis scripts to path
    addpath(genpath('~/Desktop/triangles/analysis/eeglab2024.2.1/'))
    addpath(genpath('../../analysis/'))
    addpath(genpath('/Users/janebaker/Desktop/triangles/analysis/'))
    dataDir = '/Users/janebaker/Desktop/triangles/taskData/real data/';

    % Load Eye & Task Data for the current participant
    load(fullfile(dataDir, ['p' num2str(pid) '.mat']));
    et = load(fullfile(dataDir, ['p' num2str(pid) '_eyeData.mat']));

    % Parse eye-tracker messages to find true block start and end timestamps
    msgs = string({et.othermessages.msg});
    start_mask = contains(msgs, "BLOCK_START");
    end_mask   = contains(msgs, "BLOCK_END");

    % Extract the numeric timestamps from the matching messages
    start_cell = regexp(msgs(start_mask), '\d+', 'match', 'once');
    end_cell   = regexp(msgs(end_mask), '\d+', 'match', 'once');

    blockStarts = str2double(start_cell);
    blockEnds   = str2double(end_cell);

    % Isolate timestamps specifically for Blocks 2 (Low) and 3 (High)
    % (Indices 6 and 7 represent blocks 2 and 3 in the overarching sequence)
    b2s = blockStarts(6); b2e = blockEnds(6);
    b3s = blockStarts(7); b3e = blockEnds(7);

    eyeData = et.data;
    % Check if data is binocular (more than 5 columns implies left and right eye data present)
    twoEye = size(eyeData, 2) > 5;

    % Extract raw X coordinates for both blocks using the timestamp masks
    b2mask_raw = (eyeData(:, 1) >= b2s & eyeData(:, 1) <= b2e);
    b3mask_raw = (eyeData(:, 1) >= b3s & eyeData(:, 1) <= b3e);

    if ~twoEye
        % Handle Monocular Data
        xRaw2 = eyeData(b2mask_raw, 2);
        xRaw3 = eyeData(b3mask_raw, 2);
        yRaw2 = eyeData(b2mask_raw, 3);
        yRaw3 = eyeData(b3mask_raw, 3);
    else
        % Handle Binocular Data: Extract X coordinates for both eyes
        X2 = eyeData(b2mask_raw, [2, 5]);
        X3 = eyeData(b3mask_raw, [2, 5]);

        % Clean zero values (lost tracking in one eye) by carrying over the valid eye's data
        X2(X2(:,1) == 0 & X2(:,2) ~= 0, 1) = X2(X2(:,1) == 0 & X2(:,2) ~= 0, 2);
        X2(X2(:,2) == 0 & X2(:,1) ~= 0, 2) = X2(X2(:,2) == 0 & X2(:,1) ~= 0, 1);
        X3(X3(:,1) == 0 & X3(:,2) ~= 0, 1) = X3(X3(:,1) == 0 & X3(:,2) ~= 0, 2);
        X3(X3(:,2) == 0 & X3(:,1) ~= 0, 2) = X3(X3(:,2) == 0 & X3(:,1) ~= 0, 1);

        % Average the two eyes to get a single X coordinate stream; zero out Y coordinates
        xRaw2 = mean(X2, 2); xRaw3 = mean(X3, 2);
        yRaw2 = zeros(size(xRaw2)); yRaw3 = zeros(size(xRaw3));
    end

    middle = mean([left, right]);
    halfRange = (right - left) / 2;

    % Loop through Block 2 (Low Uncertainty) and Block 3 (High Uncertainty)
    for b = 1:length(blocks)

        % Select the appropriate eye data and task data for the current block
        if b == 1
            ed_full = eyeData(b2mask_raw, :);
            td = data([data.blockNumber] == 2);
            xR = xRaw2; yR = yRaw2;
            distStd = 480;
        else
            ed_full = eyeData(b3mask_raw, :);
            td = data([data.blockNumber] == 3);
            xR = xRaw3; yR = yRaw3;
            distStd = 800;
        end
        % Downsample the full eye data based on the predetermined dsFactor
        ed_ds = ed_full(1:50:numMs, :);

        % --- FIX: POPULATE beliefShift FIELD ---
        % Identify trials where the participant changed their response key
        keys = [td.keyPressed];
        shifts = [false, diff(keys) ~= 0];
        for i = 1:length(td)
            td(i).beliefShift = shifts(i);
        end

        % Blink/Saccade Filter Logic
        % Identify blinks (pupil size/status == 0)
        if ~twoEye, isBlink = ed_ds(:, 4) == 0;
        else, isBlink = ed_ds(:, 4) == 0 | ed_ds(:, 7) == 0; end

        % Detect saccades using velocity/acceleration thresholds
        SACs = findSaccadesWithPlots(xR, yR, 'vi_min', 3.0, 'vp_min', 8.0, 'd_min', 50, 'a_min', 1.0, 'pad', 50);
        isSacRaw = false(length(xR), 1);

        % Create a boolean mask of all raw samples that fall within a saccade
        for i = 1:size(SACs, 1)
            on = SACs(i, 1); off = min(on + SACs(i, 2) - 1, length(isSacRaw));
            isSacRaw(on:off) = true;
        end

        % Apply downsampling to the raw X and Saccade mask
        xP = xR(1:50:numMs); isSac = isSacRaw(1:50:numMs);

        % Remove blinks and saccades by setting them to NaN
        filtered = xP; filtered(isSac | isBlink) = nan;

        % Normalize gaze so that -1 is the Left target and +1 is the Right target
        nFiltered = (filtered - middle) / halfRange;

        % get relevant block values
        targetPos = [td.targetX];
        currentSource = [td.sourceIsLeft];
        currSrcPlt = zeros(1, length(td));

        belief = [td.keyPressed];

        [tmpBelief, ~] = agentBelief(td, subj_Hs(p, b), left, right, distStd);

        currBlfPlt = zeros(1, length(td));

        yl = [-1 1];
        for t = 1:length(td)
            if currentSource(t)
                currSrcPlt(t) = yl(1) - 0.5;
            else
                currSrcPlt(t) = yl(2) + 0.5;
            end

            if belief(t) == 'F'
                currBlfPlt(t) = yl(1) - 0.7;
            elseif belief(t) == 'J'
                currBlfPlt(t) = yl(2) + 0.7;
            end
        end

        % normalize
        nTargetPos = (targetPos - middle) / halfRange;
        nAgentBelief = (tmpBelief - mean(tmpBelief, 'omitnan')) / std(tmpBelief, 'omitnan');
        
        % create figure 
        fig = figure;
        hold on

        y_lims = [-1.8, 1.8]; % Define the vertical span of the shading

        lb = [0.6 0.8 1];
        h1 = plot(idxs, nTargetPos, 'LineWidth', 0.5, 'Color', lb);
        h2 = plot(idxs, nFiltered, 'o', 'markerSize', 3, 'Color', 'k');

        h3 = plot(idxs, currSrcPlt, 'o', 'MarkerSize', 3);
        h4 = plot(idxs, currBlfPlt, 'o', 'MarkerSize', 3);
        h5 = plot(idxs, nAgentBelief, 'Linewidth', 1, 'Color', 'b');

        yline(middle, '--', 'Color', 'k', 'LineWidth', 1)
        yline((left - middle)/halfRange, '-', 'Color', 'k', 'LineWidth', 1)
        yline((right - middle)/halfRange, '-', 'Color', 'k', 'LineWidth', 1)

        xlim([0 6000])
        ylim(y_lims)

        legend([h1 h2 h3 h4 h5], {'Target Position', ...
            'Fixations', 'Source', ...
            'Participant Belief', 'Agent Belief'}, 'Location', 'northeast');

    end
end
%}