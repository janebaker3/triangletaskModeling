function e = objectiveFunction(params, data, eyePos, left, right, distStd)
H_subj = params(1);
m = params(2); 
b = params(3); 

% Binary vector: 1 for Right ('J'), 0 for Left ('F')
beliefRight = ([data.keyPressed] == 'J').';

% Cumulative LLR (positive = Right, negative = Left)
[tmpBelief, ~] = agentBelief(data, eyePos, H_subj, m, b, ...
    left, right, distStd);

% Sigmoid transform: maps positive LLRs to high probability of Right
choiceProbRight = 1 ./ (1 + exp(-tmpBelief));

% Bound probabilities to avoid log(0)
epsilon = 1e-10; 
choiceProbRight = max(min(choiceProbRight, 1 - epsilon), epsilon);

% Binary cross-entropy loss comparing predicted vs. actual choices
e = -sum((1 - beliefRight) .* log(1 - choiceProbRight) + beliefRight .* log(choiceProbRight));
end
