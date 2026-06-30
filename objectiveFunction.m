%% objective function to be minimized: cross-entropy error
function e = objectiveFunction(params, data, left, right, distStd)
H_subj = params(1);

% Extract participant's choices
beliefLeft = ([data.keyPressed] == 'F').';

% Run the ideal observer (agent) model to calculate Bayesian belief updates
[tmpBelief, ~] = agentBelief(data, H_subj, left, right, distStd);

% Convert absolute LLR into the probability of the favored choice
choiceProb = 1 ./ (1 + exp(tmpBelief));

epsilon = 1e-10; 
choiceProb = max(min(choiceProb, 1 - epsilon), epsilon);

e = -sum((1 - beliefLeft) .* log(1 - choiceProb) + beliefLeft .* log(choiceProb));
end
