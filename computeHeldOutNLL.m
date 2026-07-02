function heldOutNll = computeHeldOutNLL(td, xP, left, right, distStd, lb, ...
    ub, params_init, options)
t = length(td);
split = round(0.67 * t);

trainIdx = false(t, 1);
testIdx = false(t, 1);

trainIdx(1:split) = true;
testIdx(split+1:end) = true;

trainingData = td(trainIdx);
trainingEye = xP(trainIdx);

objFuncTrain = @(params) objectiveFunction(params, trainingData, trainingEye, ...
    left, right, distStd);

[fitted_params, ~] = fmincon(objFuncTrain, params_init, ...
    [], [], [], [], lb, ub, [], options);

[belief, ~] = agentBelief(td, xP, fitted_params(1), fitted_params(2), ...
    fitted_params(3), left, right, distStd); 

pRight = 1 ./ (1 + exp(-belief));
epsilon = 1e-10;
pRight = max(min(pRight, 1-epsilon), epsilon);

beliefRight = ([td.keyPressed] == 'J').';

heldOutNll = -sum(beliefRight(testIdx) .* log(pRight(testIdx)) + ...
    (1 - beliefRight(testIdx)) .* log(1 - pRight(testIdx)));

end