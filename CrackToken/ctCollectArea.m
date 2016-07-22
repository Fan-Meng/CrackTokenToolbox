function [ areaSet ] = ctCollectArea( opts, train )
%% This function prepares edge descriptor for SVM
% OUTPUTS
%  areaSet        - generated area set with following fields
%   .areaDesc      - token label histograms and edge label for each area
%   .areaInfo      - information for each area including BoundingBox,
%                    original image and imageID
%   .areaIndex     - area Id and corresponding image Id
%   .imageInfo     - necessary information of each image: Id, area
%                    number in it and size (h and w).

dfs={'modelDir','models/', 'modelFnm','crackTokenFull-150.mat', 'dataDir','../data/','reduceThr',0.40, 'histMethod','count','diskSize',4,'areaThr',500};
opts = getPrmDflt(opts,dfs,0);
load(fullfile(opts.modelDir,'forest', opts.modelFnm));
% location of ground truth
if train == 1
    imgDir = [opts.dataDir '/images/train/'];
    gtDir = [opts.dataDir '/groundTruth/train/'];
    outFile = ['areaSet/train-' opts.modelFnm '-' num2str(opts.reduceThr) '-reduce.mat'];
    if exist(outFile,'file')
        load(outFile);
        return;
    end
else
    imgDir = [opts.dataDir '/images/test/'];
    gtDir = [opts.dataDir '/groundTruth/test/'];
    outFile = ['areaSet/test-' opts.modelFnm '-' num2str(opts.reduceThr) '-reduce.mat'];
    if exist(outFile,'file')
        load(outFile);
        return;
    end
end

imgIds = dir([imgDir '*.jpg']);
imgIds = {imgIds.name};
nImgs = length(imgIds);
% nImgs =5; % for test
areaDesc = [];% histograms for each area
areaInfo = [];
index = [];
imageInfo = struct('imId','','nArea','','height','','width','');
reduceThr = opts.reduceThr;
stPath = ['../results/sketchToken/' opts.modelFnm '/']; % 输出根据crack训练出模型的crack�?��结果
if ~exist(stPath,'dir')
    mkdir(stPath);
end
for i=1:nImgs
    fprintf('Collecting edge areas from %.3d th image......\n',i);
    imgIds{i}=imgIds{i}(1:end-4);
    I = imread([imgDir imgIds{i} '.jpg']);
    gt = load([gtDir imgIds{i} '.mat']);
    gt = gt.groundTruth;
    b = gt.Boundaries;
    st = stDetect( I, model );
    E = stToEdges( st, 1 );
    if ~exist([stPath imgIds{i} '.png'],'file')
        imwrite(E,[stPath imgIds{i} '.png'],'png');
    end
    imageInfo(i,1).originalMap = E;
    Se = strel('disk',opts.diskSize);
    E2 = imclose(E,Se);
    E3 = E2;
    E3(E3<reduceThr)=0;
    imageInfo(i,1).reducedMap = E3;
    % re = bwconncomp(E3); %test for bwconncomp
    [L,num] = bwlabel(E3); % compute connective components in E3, num- number of components
    stat = regionprops(L, 'all'); % compute propoertyies of region L
    %% compute histograms for area
    [h,w,~] = size(E);
    tokenLabel = zeros(h,w,2); % 1-d.:token label for each point; 2-d. probability of this token
    %     svmgt = zeros(h,w);
    ntokens = model.opts.nClusters;
    edgeHist = zeros(num,ntokens+1);% initialize the histograms for this image
    imageEdgeHist = zeros(1,ntokens+1);
    % pE = 1-st(:,:,end);
    % choose one to generate histograms
    [tokenLabel(:,:,2),tokenLabel(:,:,1)] = max(st(:,:,1:ntokens),[],3); % lookup the most likely token
    %     [tokenLabel(:,:,2),tokenLabel(:,:,1)] = max(st(:,:,1:end),[],3); % lookup the most likely token
    %     [tokenLabel(:,:,2),tokenLabel(:,:,1)] = max(st(:,:,2:ntokens),[],3); % lookup the most likely token except 1
    % compute histogram of tokens in image
    count = hist(tokenLabel(:,:,1),1:1:ntokens+1);
    imageEdgeHist(1:1:end) = sum(count,2)';
    %     figure(1)
    %     bar(imageEdgeHist);
    %     print(['imagehist/' num2str(i)],'-depsc',figure(1));
    % compute histograms for edge areas
    bcount = zeros(num,1);%boundary point number
    bp = zeros(num,1);%boundary point porportion
    class = zeros(num,1);
    for k = 1 : num
        [y,x] = find(L==k);
        % compute center of each connected area
        %         ybar = mean(y);
        %         xbar = mean(x);
        %         text(xbar,ybar,num2str(i),'color','w');
        len = length(y);
        for j = 1 : len
            edgeHist(k,tokenLabel(y(j),x(j),1)) = edgeHist(k,tokenLabel(y(j),x(j),1))+1;
            bcount(k) = bcount(k) + b(y(j),x(j));
        end
        sumhist = sum(edgeHist(k,:));
        switch opts.histMethod
            case 'propotion'
                if (sumhist~=0) % normlize the edgeHist for each area
                    edgeHist(k,:) = edgeHist(k,:)/(sum(edgeHist(k,:)));
                end
            case 'count'
        end
        bp(k) =  bcount(k)/len;
        class(k) = bcount(k)>0 & stat(k,1).Area >= opts.areaThr;
        %         bar(edgeHist(k,:));
        %         print(figure(1),'-dpng',['hist\' num2str(k) '.png']);
    end
%     class = bcount>0 & stat(k,1).Area >= opts.areaThr;
    localData = [edgeHist class];
    areaDesc = [areaDesc;localData];
    imageInfo(i,1).imId = imgIds{i};
    imageInfo(i,1).nArea = num;
    imageInfo(i,1).height = h;
    imageInfo(i,1).width = w;
    for k = 1 : num
        local_edge_info.BoundingBox = floor(stat(k,1).BoundingBox);
        local_edge_info.Image = stat(k,1).Image;
        local_edge_info.ImId = imgIds{i};
        local_edge_info.area = stat(k,1).Area;
        %         BoundingBox = localedge.BoundingBox;
        %         if class(k)==1
        %             svmgt(BoundingBox(2)+1:BoundingBox(2)+BoundingBox(4),BoundingBox(1)+1:BoundingBox(1)+BoundingBox(3)) = localedge.Image;
        %         end
        areaInfo = [areaInfo;local_edge_info];
        index = [index; [str2double(imgIds{i}),k]];
    end
    %     figure, subplot(1,2,1),
    %     imshow(E3);
    %     colormap jet;
    %     subplot(1,2,2),
    %     imshow(svmgt);
end
areaSet.areaDesc = areaDesc;
areaSet.areaInfo = areaInfo;
areaSet.areaIndex = index;
areaSet.imageInfo = imageInfo;
save(outFile, 'areaSet');
end
