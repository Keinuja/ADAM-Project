% Simple PCA workflow for wind turbine data

clearvars
close all
clc

%% Load data
path = 'data.xlsx';
WT2  = readmatrix(path, Sheet=1, NumHeaderLines=1);
%WT14 = readmatrix(path, Sheet=3, NumHeaderLines=1);
WT39 = readmatrix(path, Sheet=4, NumHeaderLines=1);

%% Clean columns
WT2(:,[12,15,end]) = [];  
%WT14(:,[12,15]) = [];
%WT14(358,:) = [];          % remove duplicate/bad row
WT39(:,[12,15]) = [];

%% Split faulty vs good
%WT14_faulty = WT14(1:357,:);
%WT14_good   = WT14(358:end,:);
WT39_faulty = WT39(1:470,:);
%WT39_good   = WT39(471:end,:);

%% Combine healthy data
WT_good = [WT2]; %WT14_good; WT39_good

%% PCA
X = zscore(WT_good);       % standardize
C = cov(X);                
[V,D] = eig(C);            
[eigs_sorted,idx] = sort(diag(D),'descend');
W = V(:,idx);              % loadings
T = X * W;                 % scores

pcs = 6;                   
PC = T(:,1:pcs);

%% Variance explained
cumvar = cumsum(eigs_sorted / sum(eigs_sorted));
fig1 = figure;
plot(cumvar,'o-','LineWidth',2)
xlabel('Principal Component')
ylabel('Cumulative Variance Explained')
title('Variance Explained')
grid on
saveas(fig1,'variance_explained.png')

%% Project faulty data
%WT14_faulty_PC = zscore(WT14_faulty) * W(:,1:pcs);
WT39_faulty_PC = zscore(WT39_faulty) * W(:,1:pcs);

count2  = size(WT2,1);
%count14 = size(WT14_good,1);
%count39 = size(WT39_good,1);

i2_end  = count2;
%i14_end = i2_end + count14;
%i39_end = i14_end + count39;

%% 2D and 3D PCA plots
fig2 = figure;

subplot(1,2,1)
plot(PC(1:i2_end,1),PC(1:i2_end,2),'g*'); hold on
%plot(PC(i2_end+1:i14_end,1),PC(i2_end+1:i14_end,2),'c*') // 'WT14 good'
%plot(PC(i14_end+1:i39_end,1),PC(i14_end+1:i39_end,2),'k*')
%plot(WT14_faulty_PC(:,1),WT14_faulty_PC(:,2),'m*') // 'WT14 faulty'
plot(WT39_faulty_PC(:,1),WT39_faulty_PC(:,2),'y*')
xlabel('PC1'); ylabel('PC2')
legend('WT2','WT39 good','WT39 faulty')
title('2D PCA Projection')

subplot(1,2,2)
plot3(PC(1:i2_end,1),PC(1:i2_end,2),PC(1:i2_end,3),'g*'); hold on
%plot3(PC(i2_end+1:i14_end,1),PC(i2_end+1:i14_end,2),PC(i2_end+1:i14_end,3),'c*')
%plot3(PC(i14_end+1:i39_end,1),PC(i14_end+1:i39_end,2),PC(i14_end+1:i39_end,3),'k*')
%plot3(WT14_faulty_PC(:,1),WT14_faulty_PC(:,2),WT14_faulty_PC(:,3),'m*')
plot3(WT39_faulty_PC(:,1),WT39_faulty_PC(:,2),WT39_faulty_PC(:,3),'y*')
xlabel('PC1'); ylabel('PC2'); zlabel('PC3')
title('3D PCA Projection')

sgtitle('2D and 3D PCA Plots')
saveas(fig2,'pca_2d3d.png')

%% Biplots
fig3 = figure;

varNames = [ ...
    "generator speed", "Grid voltage", "Mean wind angle / s", "Average wind speed / s", ...
    "sum of generator electric quantity", "setting value of generator active power", ...
    "grid frequency", "average generator power / s", "average generator speed / s", ...
    "grid current", "engine room to north angle", "averagepitch angle / s", "reactive power", ...
    "Gen speed setpoint", "pitch angle setpoint", "Vib Y", "Vib Z", "Vib Y filtered", ...
    "Vib Z filtered", "Blade 1 temp", "Blade 2 temp", "Blade 3 temp", "Gear oil temp", ...
    "Gearbox DE bearing temp", "Gearbox NDE bearing temp", "Generator DE bearing temp", ...
    "Generator NDE bearing temp", "Stator winding U", "Stator winding V", "Stator winding W", ...
    "Hub temperature" ...
];


sgtitle('Biplots for each turbine')

subplot(1,2,1)
biplot(W(:,1:2),'Scores',T(1:i2_end,1:2), 'VarLabels', varNames(1:25))
title('WT2')

% subplot(1,3,2)
% biplot(W(:,1:2),'Scores',[T(i2_end+1:i14_end,1:2);WT14_faulty_PC(:,1:2)], 'VarLabels', varNames(1:25))
% title('WT14')

subplot(1,2,2)
biplot(W(:,1:2),'Scores', WT39_faulty_PC(:,1:2), 'VarLabels', varNames(1:25)) %T(i14_end+1:i39_end,1:2);
title('WT39')

saveas(fig3,'biplots.png')

%% Loading plots
fig4 = figure;
bar(W(:,1:3))
xlabel('Sensor')
ylabel('Loading')
legend('PC1','PC2','PC3')
grid on
title('Sensor Loadings')
