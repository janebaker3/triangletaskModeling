% Define participant pool
participantIDs = 6:13;
blocks = [2, 3];
stds = [480, 800];

% Preallocate arrays for the THREE fitted parameters
subj_Hs   = zeros(length(participantIDs), length(blocks));
subj_ms   = zeros(length(participantIDs), length(blocks));
subj_bs   = zeros(length(participantIDs), length(blocks));

numBeliefSwitches = zeros(length(participantIDs), length(blocks));
numSourceSwitches = zeros(length(participantIDs), length(blocks));
agentPart = zeros(length(participantIDs), length(blocks));
agentTask = zeros(length(participantIDs), length(blocks));

% model comparison
nll = zeros(length(participantIDs), length(blocks));
heldoutNLL = zeros(length(participantIDs), length(blocks));
avgResiduals = zeros(length(participantIDs), length(blocks));

left = 240;
right = 1680;

% Bounds for [H, w, beta]
lb = [0.000, -2, 0]; %%%%%
ub = [0.999, 2, 3]; %%%%%
k = 3; % number of free parameters
n = 6000;

numFrames = 6000;
dsFactor = 50;
numMs = numFrames * dsFactor;

% fmincon options (good practice to suppress output so it runs cleaner)
options = optimoptions('fmincon', 'Display', 'off');

dataDir = '/Users/janebaker/Desktop/triangles/taskData/real data/';

% --- Disagreement analyses ---
allConf = cell(1, length(blocks));
allDisagree = cell(1, length(blocks));
allGazeLeft = cell(1, length(blocks));
allPartLeft = cell(1, length(blocks));
allAgentLeft = cell(1, length(blocks));
allSourceLeft = cell(1, length(blocks));
allBeliefShift = cell(1, length(blocks));
allSourceShift = cell(1, length(blocks));

win = -20:20;   % window around switches
partSwitchMat = cell(1, length(blocks));
sourceSwitchMat = cell(1, length(blocks));

% store subject/block summaries if you want
meanConf_disagree = nan(length(participantIDs), length(blocks));
meanConf_agree    = nan(length(participantIDs), length(blocks));

gazeAlign_participant = nan(length(participantIDs), length(blocks));
gazeAlign_agent       = nan(length(participantIDs), length(blocks));
gazeAlign_source      = nan(length(participantIDs), length(blocks));

for p = 1:length(participantIDs)
    pid = participantIDs(p);

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

    % Calculate spatial references based on target locations
    leftMean = data(1).centers;
    rightMean = data(2).centers;

    for b = 1:length(blocks)
        distStd = stds(b);
        if b == 1
            ed_full = eyeData(b2mask_raw, :);
            td = data([data.blockNumber] == 2);
            xR = xRaw2; yR = yRaw2;
        else
            ed_full = eyeData(b3mask_raw, :);
            td = data([data.blockNumber] == 3);
            xR = xRaw3; yR = yRaw3;
        end

        % Downsample the full eye data based on the predetermined dsFactor
        ed_ds = ed_full(1:50:numMs, :);

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
        thisBlockTask = data([data.blockNumber] == blocks(b));

        %% Optimization
        % Initial guesses (true H = 0.003)
        params_init = [0.003, 0, 1]; %%%%%

        objFunc = @(params) objectiveFunction(params, thisBlockTask, xP, ...
            left, right, distStd);

        % Run fmincon
        [fitted_params, min_error] = fmincon(objFunc, params_init, ...
            [], [], [], [], lb, ub, [], options);

        % Store fitted parameters
        hFit = fitted_params(1);
        mFit = fitted_params(2);
        bFit = fitted_params(3);

        subj_Hs(p, b)   = hFit;
        subj_ms(p, b)   = mFit;
        subj_bs(p, b)   = bFit;

        % store nlls
        nll(p, b) = min_error;
        heldoutNLL(p, b) = computeHeldOutNLL(thisBlockTask, xP, left, right, distStd, lb, ...
            ub, params_init, options);

        %% count switches
        beliefLeft = [thisBlockTask.keyPressed] == 'F';
        sourceLeft = [thisBlockTask.sourceIsLeft];

        numBeliefSwitches(p, b) = sum(abs(diff(beliefLeft)));
        numSourceSwitches(p, b) = sum(abs(diff(sourceLeft)));

        %% model accuracy
        [tmpBelief, llr_seq] = agentBelief(thisBlockTask, xP, hFit, ...
            mFit, bFit, left, right, distStd);  %%%%%
        tmpBelief = tmpBelief.';
        agentLeft = (tmpBelief < 0);
        agentPart(p, b) = sum(beliefLeft == agentLeft) / length(agentLeft);
        agentTask(p,b) = sum(sourceLeft == agentLeft) / length(agentLeft);

    end
end

aic = 2 * k + 2 * nll;
bic = k * log(n) + 2 * nll;



colors = ['b', 'r'];

fig = figure;
for b = 1:length(blocks)
    h = subj_Hs(:, b);
    s = numBeliefSwitches(:, b);

    plot(h, s, 'o', 'MarkerFaceColor', colors(b), 'MarkerEdgeColor', colors(b))
    hold on
end

xlim([0 0.0055])
ylim([10 40])
xlabel('Subjective Hazard Rate')
ylabel('Number of Belief Switches')


for b = 1:length(blocks)
    m = subj_ms(:, b);

    plot(m, 'o', 'MarkerFaceColor', colors(b), 'MarkerEdgeColor', colors(b))
    hold on
end

ylim([-0.01 0.01])
close all


%% plot normalized LLR with target location, participant belief
idxs = 1:numFrames;

%% plotting llr with participant belief, target location, fixation location c
% Loop through each participant to extract and process eye-tracking and task data
for p = 1:length(participantIDs)
    pid = participantIDs(p);

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
    normalizeX = @(x) (x - middle) / halfRange;


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
        nFiltered = normalizeX(filtered);

        % get relevant block values
        targetPos = [td.targetX];
        currentSource = [td.sourceIsLeft];
        currSrcPlt = zeros(1, length(td));

        belief = [td.keyPressed];
        partLeft = (belief == 'F')';
        beliefMean = zeros(length(td), 1);
        beliefMean(find(partLeft)) = left;
        beliefMean(find(~partLeft)) = right;

        [tmpBelief, ~] = agentBelief(td, xP, subj_Hs(p, b), subj_ms(p, b), ...
            subj_bs(p,b), left, right, distStd);

        agentLeft = (tmpBelief < 0);
        disagree = ~(agentLeft == partLeft);

        % make sure shapes are columns
        tmpBelief    = tmpBelief(:);
        agentLeft    = agentLeft(:);
        partLeft     = partLeft(:);
        currentSource = currentSource(:);
        filtered     = filtered(:);

        % confidence = abs(agent decision variable)
        conf = abs(tmpBelief);

        % participant belief switches
        beliefChars = [td.keyPressed]';
        beliefShift = [false; beliefChars(2:end) ~= beliefChars(1:end-1)];

        % true source switches
        sourceShift = [false; currentSource(2:end) ~= currentSource(1:end-1)];

        % gaze side: left if gaze is left of screen midpoint
        gazeLeft = filtered < middle;

        % remove NaN gaze trials for gaze-based summaries
        validGaze = ~isnan(gazeLeft);

        % -----------------------------
        % 1. disagreement vs confidence
        % -----------------------------
        meanConf_disagree(p,b) = mean(conf(disagree), 'omitnan');
        meanConf_agree(p,b)    = mean(conf(~disagree), 'omitnan');

        % append trialwise data across all participants/blocks
        allConf{b}         = [allConf{b}; conf];
        allDisagree{b}     = [allDisagree{b}; disagree];
        allGazeLeft{b}     = [allGazeLeft{b}; gazeLeft];
        allPartLeft{b}     = [allPartLeft{b}; partLeft];
        allAgentLeft{b}    = [allAgentLeft{b}; agentLeft];
        allSourceLeft{b}   = [allSourceLeft{b}; currentSource];
        allBeliefShift{b}  = [allBeliefShift{b}; beliefShift];
        allSourceShift{b}  = [allSourceShift{b}; sourceShift];

        % ---------------------------------------------
        % 2. disagreement around participant switches
        % ---------------------------------------------
        switchIdx = find(beliefShift);
        tmpMat = nan(length(switchIdx), length(win));

        for i = 1:length(switchIdx)
            idx = switchIdx(i) + win;
            valid = idx >= 1 & idx <= length(disagree);
            tmpMat(i, valid) = disagree(idx(valid));
        end

        if ~isempty(tmpMat)
            partSwitchMat{b} = [partSwitchMat{b}; tmpMat];
        end

        % ----------------------------------------
        % 3. disagreement around true source shifts
        % ----------------------------------------
        switchIdx = find(sourceShift);
        tmpMat = nan(length(switchIdx), length(win));

        for i = 1:length(switchIdx)
            idx = switchIdx(i) + win;
            valid = idx >= 1 & idx <= length(disagree);
            tmpMat(i, valid) = disagree(idx(valid));
        end

        if ~isempty(tmpMat)
            sourceSwitchMat{b} = [sourceSwitchMat{b}; tmpMat];
        end

        % ----------------------------------------------------------
        % 4. on disagreement trials, does gaze align with participant,
        %    agent, or true source?
        % ----------------------------------------------------------
        validDisagreeGaze = disagree & validGaze;

        if any(validDisagreeGaze)
            gazeAlign_participant(p,b) = mean(gazeLeft(validDisagreeGaze) == partLeft(validDisagreeGaze));
            gazeAlign_agent(p,b)       = mean(gazeLeft(validDisagreeGaze) == agentLeft(validDisagreeGaze));
            gazeAlign_source(p,b)      = mean(gazeLeft(validDisagreeGaze) == currentSource(validDisagreeGaze));
        end

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
        nTargetPos = normalizeX(targetPos);
        nAgentBelief = (tmpBelief - mean(tmpBelief, 'omitnan')) / std(tmpBelief, 'omitnan');

        choiceProbLeft = 1 ./ (1 + exp(tmpBelief));
        yLeft = double(partLeft);   % 1 if left, 0 if right
        residual = yLeft - choiceProbLeft;
        avgResiduals(p, b) = mean(abs(residual));

        % calculate difference between gaze and current belief
        gazeSourceDiff = abs(filtered - beliefMean);
        nGSD = normalize(gazeSourceDiff);
        simpGSD = zeros(length(td), 1);
        simpGSD(find(gazeSourceDiff > middle)) = right - 100;
        simpGSD(find(gazeSourceDiff < middle)) = left + 100;

        %% create figure
        % fig = figure;
        % subplot(2, 1, 1)
        % hold on
        %
        % y_lims = [-1.8, 1.8]; % Define the vertical span of the shading
        %
        % lb = [0.6 0.8 1];
        % h1 = plot(idxs, nTargetPos, 'LineWidth', 0.5, 'Color', lb);
        % h2 = plot(idxs, nFiltered, 'o', 'markerSize', 3, 'Color', 'k');
        %
        % h3 = plot(idxs, currSrcPlt, 'o', 'MarkerSize', 3);
        % h4 = plot(idxs, currBlfPlt, 'o', 'MarkerSize', 3);
        % h5 = plot(idxs, tanh(nAgentBelief), 'Linewidth', 2, 'Color', 'b');
        % % h5 = plot(idxs, nAgentBelief, 'Linewidth', 3, 'Color', 'b');
        % % h6 = plot(idxs, nGSD, 'Linewidth', 2, 'Color', 'r');
        %
        %
        % yline(0, '--', 'Color', 'k', 'LineWidth', 1)
        % yline(normalizeX(left), '-', 'Color', 'k', 'LineWidth', 1)
        % yline(normalizeX(right), '-', 'Color', 'k', 'LineWidth', 1)
        %
        % xlim([0 6000])
        % ylim(y_lims)
        %
        % % legend([h1 h2 h3 h4 h5], {'Target Position', ...
        % %     'Fixations', 'Source', ...
        % %     'Participant Belief', 'Agent Belief'}, 'Location', 'northeast');
        %
        % subplot(2, 1, 2)
        % hold on
        % yl2 = [-0.1 1800];   % temporary/fixed y-limits for shading; adjust if needed
        % for t = 1:length(disagree)
        %     if disagree(t)
        %         patch([idxs(t)-0.5 idxs(t)+0.5 idxs(t)+0.5 idxs(t)-0.5], ...
        %             [yl2(1) yl2(1) yl2(2) yl2(2)], ...
        %             [1 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
        %     end
        % end
        %
        % plot(idxs, gazeSourceDiff, 'Linewidth', 1, 'Color', 'r');
        % yline(0, '--', 'Color', 'k', 'LineWidth', 1)
        % xlabel('frame')
        % ylabel('residual (belief mean - gaze location')
        % ylim([-0.1 1800])
    end
end

% disagreement vs confidence
figure;
for b = 1:length(blocks)
    subplot(1,2,b)
    edges = linspace(min(allConf{b}), max(allConf{b}), 12);
    [~,~,bin] = histcounts(allConf{b}, edges);

    pDisagree = nan(length(edges)-1,1);
    binCtrs = (edges(1:end-1) + edges(2:end))/2;

    for i = 1:length(pDisagree)
        pDisagree(i) = mean(allDisagree{b}(bin==i), 'omitnan');
    end

    plot(binCtrs, pDisagree, '-o', 'LineWidth', 2)
    xlabel('|Agent belief|')
    ylabel('P(disagreement)')
    title(sprintf('Disagreement vs agent confidence - Block %d', b))
    ylim([0 1])
    grid on
end

% disagreement and belief switches
figure;
for b = 1:length(blocks)
    subplot(1,2,b)
    plot(win, mean(partSwitchMat{b}, 1, 'omitnan'), 'LineWidth', 2)
    xlabel('Trials relative to participant switch')
    ylabel('P(disagreement)')
    title(sprintf('Disagreement around participant belief switches - Block %d', b))
    ylim([0 1])
    grid on
end

% disagreement and source switches
figure;
for b = 1:length(blocks)
    subplot(1,2,b)
    plot(win, mean(sourceSwitchMat{b}, 1, 'omitnan'), 'LineWidth', 2)
    xlabel('Trials relative to source switch')
    ylabel('P(disagreement)')
    title(sprintf('Disagreement around true source switches - Block %d', b))
    ylim([0 1])
    grid on
end

% does gaze align w agent or participant or source on disagreement
figure;
for b = 1:length(blocks)
    subplot(1,2,b)
    vals = [
        mean(gazeAlign_participant(:,b), 'omitnan'), ...
        mean(gazeAlign_agent(:,b), 'omitnan'), ...
        mean(gazeAlign_source(:,b), 'omitnan')
        ];

    bar(vals)
    set(gca, 'XTickLabel', {'Participant', 'Agent', 'Source'})
    ylabel('P(gaze side matches)')
    title(sprintf('On disagreement trials, what does gaze align with? - Block %d', b))
    ylim([0 1])
    grid on
end

plotMat = [meanConf_disagree(:,1), meanConf_agree(:,1), ...
           meanConf_disagree(:,2), meanConf_agree(:,2)];

figure;
bar(mean(plotMat, 1, 'omitnan'), 'FaceColor', [0.8 0.8 0.8]);
hold on

% connect each participant across the 4 conditions
for s = 1:size(plotMat,1)
    plot(1:4, plotMat(s,:), '-o', ...
        'Color', [0.4 0.4 0.4], ...
        'MarkerFaceColor', 'k', ...
        'MarkerEdgeColor', 'k');
end

set(gca, 'XTick', 1:4, ...
         'XTickLabel', {'Block1 Disagree', 'Block1 Agree', ...
                        'Block2 Disagree', 'Block2 Agree'});
ylabel('Mean |Agent Belief|')
title('Confidence for agree vs disagree trials by participant')
box off
%}