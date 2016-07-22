function [ result ] = voteSvmPredict(label, data, model)

nModel = numel(model);

for i = 1 : nModel
    [ predicted_label(:,i), ~, decision_values(:,i) ] = svmpredict( label, data, model{i,1});
end

result.countVote = sign(mean(predicted_label,2)-0.5);
result.decision_value = mean(decision_values,2);
result.nModel = nModel;