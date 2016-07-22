clear
clc
% Pablo Arbelaez <arbelaez@eecs.berkeley.edu>
% Revised by Fan Meng <im_feixie@163.com>
% description of eval files:
% 'eval_bdry_img.txt': Best threshold(T) for each image and corresponding R P F
%                       FORMAT -- [i bestT bestR bestP bestF]
% 'eval_bdry_thr.txt': Every threshold(T) and corresponding R P F for overall test set
%                       FORMAT -- [thresh R P F]
% 'eval_bdry.txt'    : Best F result with corresponding T P R and maxR maxP
%                       and maxF, and the area under PR Curve(Area_PR)
%                       FORMAT -- [bestT,bestR,bestP,bestF,R_max,P_max,F_max,Area_PR]
method = 'crackToken';

switch method
    case 'crackToken'
        path = '../eval/crackToken/';
        evalName = 'crackTokenResult';
        dirList = dir([path 'crackTokenFull*']);
        nResults = numel(dirList);
        eval = struct('evalDir','','folder','','modelName','','reduceThr','','prValues','','ODS','','OIS','');
        table_all = [];
        if ~exist(fullfile(path,[evalName '.mat']),'file')
            for i = 1 : nResults
                evalDir= fullfile(path,dirList(i).name);
                eval(i,1).evalDir = evalDir;
                eval(i,1).folder = dirList(i).name;
                tt = regexp(dirList(i).name,'-');
                nCluser = dirList(i).name(tt(1)+1:tt(2)-1);
                eval(i,1).modelName = dirList(i).name(1:tt(2)-1);
                eval(i,1).reduceThr = dirList(i).name(tt(2)+1:end);
                eval(i,1).prValues = dlmread(fullfile(evalDir,'eval_bdry_thr.txt'));
                eval(i,1).bestResult = dlmread(fullfile(evalDir,'eval_bdry.txt'));
                eval(i,1).OIS = dlmread(fullfile(evalDir,'eval_bdry_img.txt'));
                local_row = [str2double(nCluser) str2double(eval(i,1).reduceThr) eval(i,1).bestResult];
                table_all = [table_all; local_row];
            end
            save(fullfile(path,evalName),'eval','table_all');
        else
            load(fullfile(path,evalName));
        end
end

%% find best Model
[best_ODS,best_ODS_index] = max(table_all(:,6));
fprintf('Best ODS\n');
display(table_all(best_ODS_index,:));

[best_OIS,best_OIS_index] = max(table_all(:,9));
fprintf('Best OIS\n');
display(table_all(best_OIS_index,:));

[best_Area_PR, best_Area_PR_index] = max(table_all(:,end));
fprintf('Best Area_PR\n');
display(table_all(best_Area_PR_index,:));

%% plot PR curve
% plot grid table
% h=figure;
title('Pavement Crack Benchmark on CrackSet');
hold on;
[p,r] = meshgrid(0.01:0.01:1,0.01:0.01:1);
F=2*p.*r./(p+r);
[C,h] = contour(p,r,F);
% map=zeros(256,3); map(:,1)=0; map(:,2)=1; map(:,3)=0; colormap(map);
box on;
grid on;
set(gca,'XTick',0:0.1:1);
set(gca,'YTick',0:0.1:1);
set(gca,'XGrid','on');
set(gca,'YGrid','on');
xlabel('Recall');
ylabel('Precision');
axis square;
axis([0 1 0 1]);

myColor = colormap;
colorI = 1:7:64;
colormap(myColor);
% plot pr curve
w = 3;
begin = 9;
for i = 1 : 9
    prvals = eval(begin+i,1).prValues;
    plot(prvals(1:end,2),prvals(1:end,3),'color',myColor(colorI(i),:),'linewidth',w);
    hold on;
end
legend()
set(gca,'linewidth',1,'fontsize',25,'fontname','Times')


% fprintf('Boundary\n');
% fprintf('ODS: F( %1.2f, %1.2f ) = %1.2f   [th = %1.2f]\n',evalRes(2:4),evalRes(1));
% fprintf('OIS: F( %1.2f, %1.2f ) = %1.2f\n',evalRes(5:7));
% fprintf('Area_PR = %1.2f\n\n',evalRes(8));
