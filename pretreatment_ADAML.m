%% --- Integrated Data Pretreatment and PCA Workflow ---
clearvars;
close all;
clc;

%% Load dataset
path = 'data.xlsx';
WT2  = readmatrix(path, 'Sheet',1, 'NumHeaderLines',1);
WT39 = readmatrix(path, 'Sheet',4, 'NumHeaderLines',1);

%% ------------------------
%% Data Pretreatment
% Remove unwanted columns
WT2(:,[12,15,end]) = [];
WT39(:,[12,15])   = [];

% Standardize (z-score)
WT2_scaled  = zscore(WT2);
WT39_scaled = zscore(WT39);

% Replace NaN with zeros
WT2_scaled(isnan(WT2_scaled))   = 0;
WT39_scaled(isnan(WT39_scaled)) = 0;

% Handle extreme values (3-sigma rule → replace with column median)
extreme_WT2  = abs(WT2_scaled) > 3;
extreme_WT39 = abs(WT39_scaled) > 3;

for i = 1:size(WT2_scaled,2)
    colMedian = median(WT2_scaled(~extreme_WT2(:,i), i));
    WT2_scaled(extreme_WT2(:,i), i) = colMedian;
end

for i = 1:size(WT39_scaled,2)
    colMedian = median(WT39_scaled(~extreme_WT39(:,i), i));
    WT39_scaled(extreme_WT39(:,i), i) = colMedian;
end

% Synchronize dataset lengths
minLength = min(size(WT2_scaled,1), size(WT39_scaled,1));
WT2_sync  = WT2_scaled(1:minLength, :);
WT39_sync = WT39_scaled(1:minLength, :);

%% ------------------------
%% Pretreatment Summaries
datasets = {'WT2', 'WT39'};
data_vars = {WT2_sync, WT39_sync};

for k = 1:2
    data = data_vars{k};
    fprintf('\n--- %s Pretreated Data Summary ---\n', datasets{k});
    fprintf('Size: %d rows x %d columns\n', size(data,1), size(data,2));
    fprintf('Mean (first 5 cols): %s\n', mat2str(mean(data(:,1:min(5,end)))));
    fprintf('Std  (first 5 cols): %s\n', mat2str(std(data(:,1:min(5,end)))));
    fprintf('Min  (first 5 cols): %s\n', mat2str(min(data(:,1:min(5,end)))));
    fprintf('Max  (first 5 cols): %s\n', mat2str(max(data(:,1:min(5,end)))));
    num_extremes = sum(abs(data) > 3);
    fprintf('Number of extreme values replaced (per col): %s\n', mat2str(num_extremes));
end

%% ------------------------
%% Visualization (Pretreated Data)

% Line plots
fig1 = figure('Position',[100 100 1200 600]);
subplot(1,2,1); plot(WT2_sync);  title('WT2 - Pretreated Data (Line Plot)');
xlabel('Sample Index'); ylabel('Scaled Value'); grid on
subplot(1,2,2); plot(WT39_sync); title('WT39 - Pretreated Data (Line Plot)');
xlabel('Sample Index'); ylabel('Scaled Value'); grid on
sgtitle('Line Plots of Pretreated Data')
saveas(fig1,'line_plots.png')

% Histograms
fig2 = figure('Position',[100 100 1200 600]);
subplot(1,2,1); histogram(WT2_sync(:),50);  title('WT2 - Histogram');
xlabel('Scaled Value'); ylabel('Frequency'); grid on
subplot(1,2,2); histogram(WT39_sync(:),50); title('WT39 - Histogram');
xlabel('Scaled Value'); ylabel('Frequency'); grid on
sgtitle('Histograms of Pretreated Data')
saveas(fig2,'histograms.png')

% Boxplots
fig3 = figure('Position',[100 100 1200 600]);
subplot(1,2,1); boxplot(WT2_sync,'Orientation','horizontal'); title('WT2 - Boxplot');
subplot(1,2,2); boxplot(WT39_sync,'Orientation','horizontal'); title('WT39 - Boxplot');
sgtitle('Boxplots of Pretreated Data')
saveas(fig3,'boxplots.png')

%% ------------------------
%% PCA Workflow

% Split WT39 into faulty vs good (example: first 470 rows as faulty)
WT39_faulty = WT39_sync(1:470,:);

% Combine healthy datasets
WT_good = [WT2_sync];   % extend with other turbines if needed

% PCA on good data
X = zscore(WT_good);
C = cov(X);
[V,D] = eig(C);
[eigs_sorted,idx] = sort(diag(D),'descend');
W = V(:,idx);           % loadings
T = X * W;              % scores

pcs = 6;
PC = T(:,1:pcs);

% Variance explained
cumvar = cumsum(eigs_sorted / sum(eigs_sorted));
fig4 = figure;
plot(cumvar,'o-','LineWidth',2)
xlabel('Principal Component'); ylabel('Cumulative Variance Explained');
title('Variance Explained'); grid on
saveas(fig4,'variance_explained.png')

% Project faulty data
WT39_faulty_PC = zscore(WT39_faulty) * W(:,1:pcs);

count2 = size(WT2_sync,1);
i2_end = count2;

% PCA scatter plots
fig5 = figure;
subplot(1,2,1)
plot(PC(1:i2_end,1),PC(1:i2_end,2),'g*'); hold on
plot(WT39_faulty_PC(:,1),WT39_faulty_PC(:,2),'y*')
xlabel('PC1'); ylabel('PC2');
legend('WT2','WT39 faulty'); title('2D PCA Projection')

subplot(1,2,2)
plot3(PC(1:i2_end,1),PC(1:i2_end,2),PC(1:i2_end,3),'g*'); hold on
plot3(WT39_faulty_PC(:,1),WT39_faulty_PC(:,2),WT39_faulty_PC(:,3),'y*')
xlabel('PC1'); ylabel('PC2'); zlabel('PC3'); title('3D PCA Projection')
sgtitle('2D and 3D PCA Plots')
saveas(fig5,'pca_2d3d.png')

% Biplots
fig6 = figure;
varNames = ["generator speed","Grid voltage","Mean wind angle / s","Average wind speed / s", ...
    "sum of generator electric quantity","setting value of generator active power", ...
    "grid frequency","average generator power / s","average generator speed / s", ...
    "grid current","engine room to north angle","averagepitch angle / s","reactive power", ...
    "Gen speed setpoint","pitch angle setpoint","Vib Y","Vib Z","Vib Y filtered", ...
    "Vib Z filtered","Blade 1 temp","Blade 2 temp","Blade 3 temp","Gear oil temp", ...
    "Gearbox DE bearing temp","Gearbox NDE bearing temp","Generator DE bearing temp", ...
    "Generator NDE bearing temp","Stator winding U","Stator winding V","Stator winding W", ...
    "Hub temperature"];

subplot(1,2,1)
biplot(W(:,1:2),'Scores',T(1:i2_end,1:2),'VarLabels',varNames(1:25))
title('WT2')

subplot(1,2,2)
biplot(W(:,1:2),'Scores',WT39_faulty_PC(:,1:2),'VarLabels',varNames(1:25))
title('WT39')
sgtitle('Biplots for each turbine')
saveas(fig6,'biplots.png')

% Loadings plot
fig7 = figure;
bar(W(:,1:3))
xlabel('Sensor'); ylabel('Loading'); legend('PC1','PC2','PC3'); grid on
title('Sensor Loadings')
saveas(fig7,'loadings.png')
