function [ result, confMatrix ] = confusionMatrix(ture_label,predicted_label)

result.TotalNumber = length(ture_label);
result.TrueNeg = size( find( predicted_label == -1 & ture_label == 0 ), 1 );
result.FalseNeg = size( find( predicted_label == -1 & ture_label == 1 ), 1 );
result.FalsePos = size( find( predicted_label == 1 & ture_label == 0 ), 1 );
result.TruePos = size( find( predicted_label == 1 & ture_label == 1 ), 1 );
result.TrueLabels = ture_label;
result.PredictLabels = predicted_label;
% % true \ predict    1             0     
%              1    crack      
%              0                  nomal
confMatrix = zeros(2,2);
confMatrix(1,1) = result.TruePos;
confMatrix(1,2) = result.FalseNeg;
confMatrix(2,1) = result.FalsePos;
confMatrix(2,2) = result.TrueNeg ;
end