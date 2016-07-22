list = dir(['*.jpg']);
if ~exist('eps','dir')
    mkdir('eps');
end
for i = 1 : numel(list)
    im = imread(list(i).name);
    figure(1);
    imshow(im);
    print(['eps\' list(i).name(1:end-4) '.eps'],'-depsc');
end