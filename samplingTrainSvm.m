function [ svmModel ] = samplingTrainSvm(label,data,nTimes)
% sampling and balenced training of svm
% only suitable for 2-class classification
ratio = 1;% ratio = nNeg:nPos put in model
label = mapminmax(label',0,1)';
posIndex = find(label == 1);
posData = data(posIndex,:);
negIndex = find(label == 0);
nPos = length(posIndex);
nNeg = length(negIndex);
if nNeg >= nPos*ratio
    nNegSampled = nPos*ratio;
else
    nNegSampled = nNeg;
end
modelLabel = [ones(nPos,1);zeros(nNegSampled,1)];
for i = 1 : nTimes
    innerIndex = randperm(nNeg,nNegSampled);
    negData = data(negIndex(innerIndex),:);
    modelData = [posData;negData];
    model = svmtrain(modelLabel,modelData);
    svmModel{i,1} = model;
end