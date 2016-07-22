function [] = showgt(name)
load([name '.mat']);
figure(1);
imshow(1-groundTruth.Boundaries);
% print('-depsc',[name '.eps']);
print('-djpeg',[name '-gt.jpg']);
end
