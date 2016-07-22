gtIds = dir('*.mat');
gtIds = {gtIds.name};
nGt = length(gtIds);

for i = 1 : nGt
    gt = load(gtIds{i});
    northWest = groundTruth.Segmentation(1,1);
    fprintf(['The north west corner is %d in ' gtIds{i} '\n'],northWest);
end