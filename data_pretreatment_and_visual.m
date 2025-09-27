%% --- Data Pretreatment Visualization with Summaries (Export PNGs) ---
clearvars;
close all;
clc;

%% Load dataset
path = 'data.xlsx';
WT2  = readmatrix(path, 'Sheet',1, 'NumHeaderLines',1);
WT39 = readmatrix(path, 'Sheet',4, 'NumHeaderLines',1);

%% Feedback incorporation / column cleaning
WT2(:,[12,15,end]) = [];
WT39(:,[12,15])     = [];

%% Centering and scaling
WT2_scaled  = zscore(WT2);
WT39_scaled = zscore(WT39);

%% Handle missing values
WT2_scaled(isnan(WT2_scaled))   = 0;
WT39_scaled(isnan(WT39_scaled)) = 0;

%% Handle extreme values (3-sigma rule)
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

%% Time synchronization
minLength = min(size(WT2_scaled,1), size(WT39_scaled,1));
WT2_sync  = WT2_scaled(1:minLength, :);
WT39_sync = WT39_scaled(1:minLength, :);

%% Summary outputs for command window
datasets = {'WT2', 'WT39'};
data_vars = {WT2_sync, WT39_sync};

for k = 1:2
    data = data_vars{k};
    fprintf('\n--- %s Pretreated Data Summary ---\n', datasets{k});
    fprintf('Size: %d rows x %d columns\n', size(data,1), size(data,2));
    fprintf('Mean (first 5 columns): %s\n', mat2str(mean(data(:,1:min(5,end)))));
    fprintf('Std  (first 5 columns): %s\n', mat2str(std(data(:,1:min(5,end)))));
    fprintf('Min  (first 5 columns): %s\n', mat2str(min(data(:,1:min(5,end)))));
    fprintf('Max  (first 5 columns): %s\n', mat2str(max(data(:,1:min(5,end)))));
    num_extremes = sum(abs(data) > 3);
    fprintf('Number of extreme values replaced (per column): %s\n', mat2str(num_extremes));
end

%% -----------------------
%% Visualization and Export as PNG

% Line plots
fig1 = figure('Position',[100 100 1200 600]);
subplot(1,2,1)
plot(WT2_sync)
title('WT2 - Pretreated Data (Line Plot)')
xlabel('Sample Index'); ylabel('Scaled Value'); grid on

subplot(1,2,2)
plot(WT39_sync)
title('WT39 - Pretreated Data (Line Plot)')
xlabel('Sample Index'); ylabel('Scaled Value'); grid on

sgtitle('Line Plots of Pretreated Data')
saveas(fig1,'line_plots.png')

% Histograms
fig2 = figure('Position',[100 100 1200 600]);
subplot(1,2,1)
histogram(WT2_sync(:),50)
title('WT2 - Histogram')
xlabel('Scaled Value'); ylabel('Frequency'); grid on

subplot(1,2,2)
histogram(WT39_sync(:),50)
title('WT39 - Histogram')
xlabel('Scaled Value'); ylabel('Frequency'); grid on

sgtitle('Histograms of Pretreated Data')
saveas(fig2,'histograms.png')

% Boxplots
fig3 = figure('Position',[100 100 1200 600]);
subplot(1,2,1)
boxplot(WT2_sync,'Orientation','horizontal')
title('WT2 - Boxplot')

subplot(1,2,2)
boxplot(WT39_sync,'Orientation','horizontal')
title('WT39 - Boxplot')

sgtitle('Boxplots of Pretreated Data')
saveas(fig3,'boxplots.png')