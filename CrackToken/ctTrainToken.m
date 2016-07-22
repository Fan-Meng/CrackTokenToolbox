function opts = ctTrainToken( varargin )
%% Training tokens is the firts step in crack detection based on sketch token
%
% dfs={'nPatches',inf,'ds',200,'dN',200,...
%     'nClusters',150, 'nTrees',25, 'radius',15, 'nPos',1000, 'nNeg',800,...
%     'negDist',2, 'minCount',4, 'nCells',5, 'normRad',5, 'normConst',0.01, ...
%     'nOrients',[4 4 0], 'sigmas',[0 1.5 5], 'chnsSmooth',2, 'fracFtrs',1, ...
%     'seed',1, 'modelDir','models/', 'modelFnm','CrackTokenFull', 'dividePath','Divide Patch Desctriptor',...
%     'clusterFnm','crack_cluster.mat', 'dataDir','../data/testData'};
dfs={'nPatches',inf,'ds',200,'dN',200,'radius',15,...
    'nClusters',150, 'nKmeanCluster',500, 'modelFnm','CrackTokenFull', 'dividePath','Divide Patch Desctriptor',...
    'clusterFnm','crack_clusters.mat', 'dataDir','../data/testData'};
opts = getPrmDflt(varargin,dfs,-1);% add extra parameter in varargin which are not in dfs
addpath(genpath('clusters'));
%% Step 1. collect patches
if exist(opts.clusterFnm,'file')
    load(opts.clusterFnm);
    return;
end
radius = opts.radius;
nPatches = opts.nPatches;
dataDir = opts.dataDir;
clusters = stGetPatches( radius, nPatches, dataDir );
% save(opts.clusterFnm,'clusters');
%% step 2. compute daisy feature for each patch
h = opts.radius*2+1;
w = opts.radius*2+1;
ds = opts.ds;
dividePath = opts.dividePath;
if ~exist(dividePath,'dir'), mkdir(dividePath);end
pn = size(clusters.patches,3);
fprintf('-------------------------------------------------------\n');
fprintf('Total number of patches is: %d\n',pn);
patches = clusters.patches;
% step 2.1 ���������ֳ������飬�ֱ����daisy
% the computation is memory comsuming, so we conduct a divide and divide
% and conquer strategy
dN = opts.dN;%Ϊ�ֿ����ĸ���
name = cell(dN,1);
for k = 1 : dN
    num = num2str(k);
    name{k} = ['pdesc' num '.mat'];
end
partN(1) = 0;
partN(2:dN) = ones(1,dN-1)*floor(pn/dN);
partN(dN+1) = pn - sum(partN(1:dN));
index = cumsum(partN);
lenPrev = 0;

for k = 1 : dN
    %     fprintf( repmat('\b', [1 lenPrev] ) )
    %     lenPrev = 51;
    fprintf('This is the %.2d th part, %.2d part(s) remaining......\n',k,dN-k);
    %     if exist(fullfile(dividePath,name{k}),'file')
    %         coutinue;
    %     end
    tic;
    pf = sparse(partN(k+1),h*w*ds);
    beginIndex = index(k);
    parfor i = index(k)+1 : index(k+1)
        patch = patches(:,:,i);
        dzy = compute_daisy(patch);
        [y,x] = find(patch);
        nlables = length(x);
        for j = 1 : nlables
            feature = zeros(1,h*w*ds);
            feature(1,((y(j)-1)*w+x(j)-1)*ds+1:((y(j)-1)*w+x(j))*ds) = dzy.descs((y(j)-1)*w+x(j),:);
        end
        pf(i-beginIndex,:) = feature(1,:);
    end
    save(fullfile(dividePath,name{k}), 'pf');
    clear pf;
    toc;
end
clear patches;

% step 2.1 ����������daisy���ӽ����кϲ�
clear pf;
pfs = sparse(pn,h*w*ds);
for k = 1 : dN
    fprintf('.....k=%d.....\n',k);
    load(fullfile(dividePath,name{k}));
    pfs(index(k)+1:index(k+1),:) = pf(1:end,:);
end
clusters.daisy = pfs;

save('crack_daisy.mat','pfs');
clear pfs pf;
%% the code above need run only once, daisy descriptors and patches are stored in 'crack_clusters.mat' as clusters.daisy

%% step 3. clustering patches to get crack token
nClusters = opts.nKmeanCluster;
[clusterId,clusterCenter] = clusterPatches(clusters.patches,clusters.daisy,nClusters);
clusters.clusterId = clusterId;
clusters.clusters = clusterCenter;
save(opts.clusterFnm,'clusters');

%% show the cluster centers
tic;
figure(1);
montage2(clusters.clusters);
% print('tokens-500.eps','-depsc');
figure(2);
montage2(clusters.clusters(:,:,1:150));
% print('tokens-150.eps','-depsc');
toc;

% %% update the cluster 1
% index_1 = find(clusters.clusterId==1);
% daisy_1 = clusters.daisy(index_1,:);
% patch_1 = clusters.patches(:,:,index_1);
% nCluster_1 = 10;
% tic
% [clusterId_1,clusterCenter_1] = clusterPatches(patch_1,daisy_1,nCluster_1);
% toc;
% montage2(clusterCenter_1);
end

function [clusterId,clusterCenter] = clusterPatches(patch,feature,nClusters)
fprintf('In process of patch clustering......\n');
tic;
clusterId = kmeans2(feature,nClusters);
toc;
%
% ccount = zeros(nCluster,2);
% ccount(:,1) = 1:1:nCluster;
%
% ��ÿ��patch�����������?
parfor i = 1 : nClusters
    ccount(i,2) = length(find(clusterId==i));
end
% ����ÿ����ľ��?
shape = size(patch);
shape(1,3) = nClusters;
clusterCenter = zeros(shape);
for i = 1 : nClusters
    members  = patch(:,:,clusterId==i);
    c = size(members,3);
    memberSum = zeros(shape(1),shape(2));
    for j = 1 : c
        memberSum = memberSum + members(:,:,j);
    end
    clusterCenter(:,:,i) = memberSum/c;
end
end
