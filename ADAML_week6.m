%% --- Integrated Data Pretreatment, PCA and modeling Workflow ---
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
%% Split into training and balanced validation
n_val = 20;

% Validation set
WT2_val   = WT2_sync(1:n_val, :);      % 20 healthy
WT39_val  = WT39_sync(1:n_val, :);     % 20 faulty
labels_val = [zeros(n_val, 1); ones(n_val, 1)]; % 0=healthy, 1=faulty

% Training set (remaining samples)
WT2_train  = WT2_sync(n_val+1:end, :);
WT39_train = WT39_sync(n_val+1:end, :);
labels_train = [zeros(size(WT2_train, 1), 1); ones(size(WT39_train, 1), 1)];

%Combined train and test values for PLS-DA
WT_train_all = [WT2_train; WT39_train];
WT_test_all = [WT2_val; WT39_val];

% Only healthy training for PCA
WT_train = WT2_train;

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
WT_faulty = WT39_sync;

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
WT_faulty_PC = zscore(WT_faulty) * W(:,1:pcs);

count2 = size(WT2_sync,1);
i2_end = count2;

% PCA scatter plots
fig5 = figure;
subplot(1,2,1)
plot(PC(1:i2_end,1),PC(1:i2_end,2),'g*'); hold on
plot(WT_faulty_PC(:,1),WT_faulty_PC(:,2),'y*')
xlabel('PC1'); ylabel('PC2');
legend('WT2','WT39 faulty'); title('2D PCA Projection')

subplot(1,2,2)
plot3(PC(1:i2_end,1),PC(1:i2_end,2),PC(1:i2_end,3),'g*'); hold on
plot3(WT_faulty_PC(:,1),WT_faulty_PC(:,2),WT_faulty_PC(:,3),'y*')
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
biplot(W(:,1:2),'Scores',WT_faulty_PC(:,1:2),'VarLabels',varNames(1:25))
title('WT39')
sgtitle('Biplots for each turbine')
saveas(fig6,'biplots.png')

% Loadings plot
fig7 = figure;
bar(W(:,1:3))
xlabel('Sensor'); ylabel('Loading'); legend('PC1','PC2','PC3'); grid on
title('Sensor Loadings')
saveas(fig7,'loadings.png')

%%----------------------------------
%% PCA Based fault detection metrics
%T^2 and SPEx

%PCA is done to the WT2 (healthy) T2 and SPE to the WT39 (faulty)

latent = eigs_sorted;
coeff = W;
score = T;

T2 = sum((score(:,1:pcs) / diag(latent(1:pcs)' )).^2, 2);

%control limit for T2
n = size(WT39_scaled, 1);
alpha = 0.95;
T2_lim = pcs * (n-1) / (n-pcs) * finv(alpha, pcs, n-pcs);

%SPEx
WT39_hat = score(:, 1:pcs) * coeff(:, 1:pcs)'; 
WT39_residual = WT39_scaled - WT39_hat;

SPEx = sum(WT39_residual.^2, 2);

lambda = latent(pcs+1:end);

th1 = sum(lambda);
th2 = sum(lambda.^2);
th3 = sum(lambda.^3);

h0 = 1 - (2 * th1 * th3) / (3 * th2^2);
z_alpha = norminv(alpha);

Q_lim = th1 * ( (z_alpha * sqrt(2 * th2 * h0^2) / th1) + 1 + ( th2 * h0 * (h0 - 1) ) / (th1^2) )^(1 / h0);

T2_fault = find(T2 > T2_lim);
SPEx_fault = find(SPEx > Q_lim);

T2_fault_start = min(T2_fault)
SPEx_fault_start = min(SPEx_fault)

%visuals
fig8 = figure;
subplot(2,1,1)
plot(T2, 'b');
grid on
hold on
plot(T2_fault, T2(T2_fault), 'g*');
yline(T2_lim, 'r--', 'T^2 Limit');
title('T^2')
xlabel('Sample')
ylabel('T^2')

subplot(2,1,2)
plot(SPEx, 'b')
grid on
hold on
%plot(SPEx(SPEx_fault), 'g*')
yline(Q_lim, 'r--', 'SPEx limit')
title('SPEx')
xlabel('Sample')
ylabel('Q')
saveas(fig8,'PCA_Metrics.png')

%% ------------------------
%% Fault detection via reconstruction error
X_train_recon = PC_train * W(:,1:pcs)'; 
train_error = mean((WT_train_scaled - X_train_recon).^2,2);

PC_val = WT_val_scaled * W(:,1:pcs);
X_val_recon = PC_val * W(:,1:pcs)';
val_error = mean((WT_val_scaled - X_val_recon).^2,2);

threshold = prctile(train_error,95);
pred_labels = double(val_error > threshold);

accuracy = sum(pred_labels == labels_val) / length(labels_val);
fprintf('Validation Accuracy: %.2f%%\n', accuracy*100);

confMat = confusionmat(labels_val, pred_labels);
disp('Confusion Matrix [Healthy; Faulty]:')
disp(confMat)

% Reconstruction error histogram
fig9 = figure;
histogram(train_error,50,'FaceColor','g'); hold on;
histogram(val_error(labels_val==1),50,'FaceColor','r');
xline(threshold,'k--','LineWidth',2);
legend('Healthy Training','Faulty Validation','Threshold');
xlabel('Reconstruction Error'); ylabel('Frequency'); grid on
title('Reconstruction Error Distribution')
saveas(fig9,'recon_error_hist.png')

%% ------------------------
%% PCA Scatter plots
fig10 = figure('Position',[100 100 1200 600]);
subplot(1,2,1)
plot(PC_train(:,1),PC_train(:,2),'g*'); hold on
plot(PC_val(labels_val==0,1),PC_val(labels_val==0,2),'bo')
plot(PC_val(labels_val==1,1),PC_val(labels_val==1,2),'r+')
xlabel('PC1'); ylabel('PC2'); title('2D PCA Projection')
legend('Healthy Training','Healthy Validation','Faulty Validation'); grid on

subplot(1,2,2)
plot3(PC_train(:,1),PC_train(:,2),PC_train(:,3),'g*'); hold on
plot3(PC_val(labels_val==0,1),PC_val(labels_val==0,2),PC_val(labels_val==0,3),'bo')
plot3(PC_val(labels_val==1,1),PC_val(labels_val==1,2),PC_val(labels_val==1,3),'r+')
xlabel('PC1'); ylabel('PC2'); zlabel('PC3'); title('3D PCA Projection'); grid on
sgtitle('PCA Projections - Training vs Validation')
saveas(fig10,'pca_scatter.png')

%% ------------------------
%% Loadings plot
fig11 = figure;
bar(W(:,1:3))
xlabel('Sensor'); ylabel('Loading'); legend('PC1','PC2','PC3'); grid on
title('Sensor Loadings')
saveas(fig11,'loadings.png')


%%-------
%% PLS-DA

%PLS da jutut
[XL, YL, XS, YS, BETA, PTCVAR, MSE, stats] = plsregress(WT_train_all, labels_train, pcs);
%BETA = regression coeff vector
%stats = weights

WT_test_aug = [ones(size(WT_test_all, 1), 1) WT_test_all];
Ypred = WT_test_aug * BETA;

Yclass = Ypred > 0.5; %treshold

%evaluation
accuracy_PLS = sum(Yclass == labels_val) / length(labels_val);

fprintf('Validation Accuracy for PLS-DA: %.2f%%\n', accuracy_PLS*100);

confMat_PLS = confusionmat(labels_val, double(Yclass));
disp('Confusion Matrix [Healthy; Faulty]:')
disp(confMat_PLS)