function model = ctTrain(varargin)
%% traing Sketch Token model
addpath(genpath('../toolbox'));
dfs={'nPatches',inf,'ds',200,'dN',200,'reduceThr',.4,'histMethod','count',...
    'nClusters',150, 'nTrees',25, 'radius',15, 'nPos',1000, 'nNeg',800,...
    'negDist',2, 'minCount',4, 'nCells',5, 'normRad',5, 'normConst',0.01, ...
    'nOrients',[4 4 0], 'sigmas',[0 1.5 5], 'chnsSmooth',2, 'fracFtrs',1, ...
    'seed',1, 'modelDir','models/', 'modelFnm','CrackTokenFull', 'dividePath','Divide Patch Desctriptor',...
    'clusterFnm','crack_cluster.mat', 'dataDir','../data/testData','diskSize',4,'test',0};
opts = getPrmDflt(varargin,dfs,1);
opts = ctTrainToken(opts);
tic, model=ctTrainModel(opts); toc

%% train SVM model for edge classification

% if ~exist(['areaSet/train_areaSet-' num2str(opts.reduceThr) '.mat'],'file')
    [ areaSet ] = ctCollectArea(opts,1);
% else
%     load(['areaSet/train_areaSet-' num2str(opts.reduceThr) '.mat']);
% end

% building svm model
training_label = areaSet.areaDesc(:,end);
training_data = areaSet.areaDesc(:,1:end-1);
[ svmModel ] = samplingTrainSvm(training_label,training_data,10);
% [ svmModel ] = ballencingTrainSvm(training_label,training_data);
% model.svmModel = svmModel;
% save(fullfile(opts.modelDir,'svm', [opts.modelFnm '-svmModel']),'svmModel');
%% test on test set
% if ~exist(['areaSet/test_areaSet-' num2str(opts.reduceThr) '.mat'],'file')
    [ areaSet ] = ctCollectArea(opts,0);
% else
%     load(['areaSet/test_areaSet-' num2str(opts.reduceThr) '.mat']);
% end

test_label = areaSet.areaDesc(:,end);
test_data = areaSet.areaDesc(:,1:end-1);
[ predictResult ] = voteSvmPredict(test_label, test_data, svmModel);
predicted_label = predictResult.countVote;

% % single model
% svmModel = svmtrain(training_label,areaSet.areaDesc(:,1:end-1));
% [ predicted_label, accuracy, decision_values ] = svmpredict( training_label, areaSet.areaDesc(:,1:end-1), svmModel );

[ result ,confMatrix ] = confusionMatrix( test_label, predicted_label );
% predict crack map based on svm result
if opts.test
    opts.outPath = fullfile('../results/test/crackToken');
    opts.method = 'crackToken-test1';
else
    opts.outPath = fullfile('../results', 'crackToken', [opts.modelFnm '-' num2str(opts.reduceThr)]);
    opts.method =['crackToken-' num2str(opts.reduceThr)];
end
[ predictedMap ] = predictMap( predicted_label, areaSet, opts );
% evaluate the total result


bench_cracks(opts);

% [ evalResults ] = crackEval(predictedMap, opts);
