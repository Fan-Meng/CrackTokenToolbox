%% demo of crack token
%% test
opts=struct('nPos',100,'nNeg',80,'modelFnm','test_0713','nTrees',5,'clusterFnm','crack_cluster_test.mat',...
    'dataDir','../data/','test',1);
model = ctTrain(opts);

% opts=struct('nPos',1000,'nNeg',800,'modelFnm','crackTokenFull-100','nTrees',25,'clusterFnm','crack_clusters.mat',...
%     'dataDir','../data/','test',0);
% model = ctTrain(opts);



%% full
% opts=struct('nPos',1000,'nNeg',800,'modelFnm','test','nTrees',25,'clusterFnm','crack_cluster_full.mat','dataDir','../data/');
% model = ctTrain(opts);
% % small model
% opts=struct('nPos',100,'nNeg',80,'modelFnm','crack_all_500_150','nTrees',20,'clusterFnm','crack_cluster.mat');
% tic, model=ctTrainToken(opts); toc
% % big model
% tic, model=ctTrainToken(opts); toc

%% eval all
modelPath = 'models/forest/';
modelList = dir([modelPath '*.mat']);
reduceThr = 0.4:0.05:0.8;
opts=struct('clusterFnm','crack_clusters.mat', 'dataDir','../data/');
for i = 1 : numel(modelList)
    opts.modelFnm = modelList(i).name(1:end-4);
    for j = 1 : numel(reduceThr)
        opts.reduceThr = reduceThr(j);
        model = ctTrain(opts);
        bench_cracks(model.opts);
    end
end
opts.method='CrackTree';
bench_cracks(opts);
