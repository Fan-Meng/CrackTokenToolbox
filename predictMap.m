function [ predictedMap ] = predictMap( preLabel, areaSet, para )
%% generate edge map based on the predicted edge area

outPath = para.outPath;
bmapPath = fullfile(outPath, 'bmap');
pmapPath = fullfile(outPath, 'pmap');

if(~exist(outPath,'dir')),  mkdir(outPath); end;
if(~exist(bmapPath,'dir')), mkdir(bmapPath); end;
if(~exist(pmapPath,'dir')), mkdir(pmapPath); end;

areaInfo = areaSet.areaInfo;
imageInfo = areaSet.imageInfo;
nImage = length(imageInfo(:,1));
predictedMap(nImage,1) = struct('image',[],'id',[]);

countA = 0;
for i = 1 : nImage
    h = imageInfo(i,1).height;
    w = imageInfo(i,1).width;
    predictedMap(i,1).bImage = zeros(h,w);% binary image
    predictedMap(i,1).id = imageInfo(i,1).imId;
    for j = 1 : imageInfo(i,1).nArea
        countA = countA + 1;          
        if preLabel(countA)==1
            BoundingBox = areaInfo(countA,1).BoundingBox;
            predictedMap(i,1).bImage(BoundingBox(2)+1:BoundingBox(2)+BoundingBox(4),BoundingBox(1)+1:BoundingBox(1)+BoundingBox(3))= areaInfo(countA,1).Image;
        end
    end
    predictedMap(i,1).pImage = areaSet.imageInfo(i,1).reducedMap;
    predictedMap(i,1).pImage(predictedMap(i,1).bImage==0)=0;
%     predictedMap(i,1).bdrypmap = seg2bdry( predictedMap(i,1).pImage, 'imagesize');
%     figure(i)
%     subplot(1,2,1), im(predictedMap(i,1).pImage);
%     subplot(1,2,2), im(predictedMap(i,1).bImage);
%     imwrite(predictedMap(i,1).bImage,[outPath num2str(imageInfo(i,1).imId) '.png'],'png');
    imwrite(predictedMap(i,1).bImage,fullfile(bmapPath, [num2str(imageInfo(i,1).imId) '.png']),'png');
    imwrite(predictedMap(i,1).pImage,fullfile(pmapPath, [num2str(imageInfo(i,1).imId) '.png']),'png');
end
end