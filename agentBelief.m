function [belief, llr_seq] = agentBelief(td, eyePos, H, m, b, ...
    left, right, distStd)

    belief    = zeros(length(td), 1);
    targetPos = [td.targetX]';

    zLeft  = normcdf(right, left,  distStd) - normcdf(left, left,  distStd);
    zRight = normcdf(right, right, distStd) - normcdf(left, right, distStd);

    probLeft  = normpdf(targetPos, left,  distStd) ./ zLeft;
    probRight = normpdf(targetPos, right, distStd) ./ zRight;

    dist = abs(eyePos - targetPos);
    w = max(m .* dist + b, 0);

    llr_raw = log(probRight ./ probLeft);
    llr_seq = w .* llr_raw;

    probSwitch = (1-H) / H;
    belief(1)  = llr_seq(1);

    for f = 2:length(td)
        prevBelief = belief(f-1);
        prior      = prevBelief ...
                   + log(probSwitch + exp(-prevBelief)) ...
                   - log(probSwitch + exp( prevBelief));
        belief(f)  = prior + llr_seq(f);
    end
end
