 list = dir('*.mat');
if ~exist('eps','dir'), mkdir('eps'); end
for i = 1 : numel(list)
    load(list(i).name);
    gt = 1 - groundTruth.Boundaries;
    imshow(gt);
    print(['eps\' list(i).name(1:end-4) '-gt.eps'],'-depsc');
end