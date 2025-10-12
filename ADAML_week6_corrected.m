%% --- Integrated Data Pretreatment, PCA, SPEx (SPE) and PLS-DA Workflow ---
% Updated, consistent end-to-end script: training on WT2 (healthy),
% unsupervised PCA/SPEx/T^2 detection on WT39 (faulty), and PLS-DA classification.
clearvars;
close all;
clc;

%% ------------------------
%% Load dataset
path = 'data.xlsx';
WT2  = readmatrix(path, 'Sheet',1, 'NumHeaderLines',1);
WT39 = readmatrix(path, 'Sheet',4, 'NumHeaderLines',1);

%% ------------------------
%% Data Pretreatment
% Remove unwanted/non-physical columns (feedback-based)
WT2(:,[12,15,end]) = [];
WT39(:,[12,15])    = [];

% Replace NaNs with column medians before scaling (safer than zeros pre-scale)
for j = 1:size(WT2,2)
    col = WT2(:,j);
    if any(isnan(col))
        col(isnan(col)) = median(col(~isnan(col)));
        WT2(:,j) = col;
    end
end
for j = 1:size(WT39,2)
    col = WT39(:,j);
    if any(isnan(col))
        col(isnan(col)) = median(col(~isnan(col)));
        WT39(:,j) = col;
    end
end

% Synchronize dataset lengths (use equal length for convenience)
minLength = min(size(WT2,1), size(WT39,1));
WT2  = WT2(1:minLength, :);
WT39 = WT39(1:minLength, :);

%% ------------------------
%% Split into training and balanced validation
n_val = 20;  % number of samples per class for validation

% --- Healthy (WT2) ---
WT2_train = WT2(1:end-n_val, :);   % all but last 20 samples for training
WT2_val   = WT2(end-n_val+1:end,:);% last 20 for validation

% --- Faulty (WT39) ---
WT39_train = WT39(1:end-n_val, :); % first samples for calibration/training
WT39_val   = WT39(end-n_val+1:end,:); % next 20 samples (later fault) for validation

labels_val = [zeros(n_val,1); ones(n_val,1)];  % 0 = healthy, 1 = faulty

% Training data for unsupervised PCA uses healthy only (WT2_train)
WT_train = WT2_train;

% Combined datasets for supervised PLS-DA
WT_train_all = [WT2_train; WT39_train];
labels_train = [zeros(size(WT2_train,1),1); ones(size(WT39_train,1),1)];
WT_test_all  = [WT2_val; WT39_val];    % test/prediction inputs for PLS-DA

%% ------------------------
%% Standardize using training (WT2_train) statistics
mu = mean(WT_train, 1);
sigma = std(WT_train, 0, 1);
sigma(sigma==0) = 1;   % avoid division by zero for constant columns

WT_train_scaled  = (WT_train - mu) ./ sigma;
WT2_val_scaled   = (WT2_val - mu) ./ sigma;
WT39_val_scaled  = (WT39_val - mu) ./ sigma;
WT_val_scaled    = [WT2_val_scaled; WT39_val_scaled];

WT39_scaled_all  = (WT39 - mu) ./ sigma;   % full faulty scaled for PCA projection
WT_train_all_scaled = (WT_train_all - mu) ./ sigma; % for supervised training
WT_test_all_scaled  = (WT_test_all - mu) ./ sigma;  % for supervised testing

%% ------------------------
%% Handle extreme values (3-sigma rule) on scaled training & validation
% Replace values with column median computed from non-extreme training values
extreme_train = abs(WT_train_scaled) > 3;
for j = 1:size(WT_train_scaled,2)
    colMed = median(WT_train_scaled(~extreme_train(:,j), j));
    if isempty(colMed) || isnan(colMed), colMed = 0; end
    WT_train_scaled(extreme_train(:,j), j) = colMed;
end

% Apply same clipping/median replacement to WT39_scaled_all and WT_val_scaled
extreme_val = abs(WT_val_scaled) > 3;
for j = 1:size(WT_val_scaled,2)
    colMed = median(WT_train_scaled(:, j)); % use training median
    idx = extreme_val(:,j);
    WT_val_scaled(idx,j) = colMed;
end
extreme_39 = abs(WT39_scaled_all) > 3;
for j = 1:size(WT39_scaled_all,2)
    colMed = median(WT_train_scaled(:, j));
    idx = extreme_39(:,j);
    WT39_scaled_all(idx,j) = colMed;
end

% Also update scaled combined sets for PLS
WT_train_all_scaled = (WT_train_all - mu) ./ sigma;
WT_test_all_scaled  = (WT_test_all - mu) ./ sigma;

%% ------------------------
%% Visualization of pretreated data (optional quick checks)
fig1 = figure('Position',[100 100 1200 600]);
subplot(1,2,1); plot(WT_train_scaled); title('WT2 Training (Scaled)'); xlabel('Sample'); ylabel('Scaled Value'); grid on;
subplot(1,2,2); plot(WT39_scaled_all); title('WT39 (Scaled)'); xlabel('Sample'); ylabel('Scaled Value'); grid on;
sgtitle('Pretreated Time Series (Training vs Faulty)');
saveas(fig1,'pretreat_lines.png');

%% ------------------------
%% PCA on healthy training (WT_train_scaled)
C = cov(WT_train_scaled);
[V,D] = eig(C);
[eigs_sorted, idx] = sort(diag(D), 'descend');
W = V(:, idx);             % loadings (columns are eigenvectors sorted by variance)
lambda_all = eigs_sorted;  % eigenvalues sorted descending

% Scores for training and set number of PCs
T_train = WT_train_scaled * W;
pcs = 6;
PC_train = T_train(:,1:pcs);

% cumulative variance explained
cumvar = cumsum(lambda_all / sum(lambda_all));
fig2 = figure;
plot(cumvar,'o-','LineWidth',2); xlabel('Principal Component'); ylabel('Cumulative Variance Explained');
title('Variance Explained - WT2 (Healthy Training)'); grid on;
saveas(fig2,'variance_explained.png');

%% ------------------------
%% Project faulty data (WT39) into PCA space using training loadings
T39 = WT39_scaled_all * W;    % full scores for WT39 (using training W)
PC39 = T39(:,1:pcs);          % first pcs scores

% Reconstruction of WT39 using first pcs
WT39_recon = PC39 * W(:,1:pcs)';     % reconstructed (using pcs)
WT39_residual = WT39_scaled_all - WT39_recon;
SPEx = sum(WT39_residual.^2, 2);     % SPE (Q-statistic) per sample

% Reconstruction for training (to compute training SPE)
WT_train_recon = PC_train * W(:,1:pcs)';
train_residual = WT_train_scaled - WT_train_recon;
SPE_train = sum(train_residual.^2, 2);

%% ------------------------
%% Hotelling's T^2 calculations (training & WT39)
lambda_pcs = lambda_all(1:pcs);     % eigenvalues for selected pcs
lambda_pcs(lambda_pcs==0) = eps;

T2_train = sum( (PC_train.^2) ./ (lambda_pcs'), 2 );   % training T^2
T2_39   = sum( (PC39.^2)   ./ (lambda_pcs'), 2 );     % WT39 T^2

% T2 control limit (using training sample size)
n_train = size(WT_train_scaled, 1);
alpha = 0.95;
T2_lim = (pcs*(n_train-1) / (n_train-pcs)) * finv(alpha, pcs, n_train-pcs);

%% ------------------------
%% Q-limit (SPE) using Jackson-Mudholkar (training residual eigenvalues)
lambda_res = lambda_all(pcs+1:end);
g1 = sum(lambda_res);
g2 = sum(lambda_res.^2);
g3 = sum(lambda_res.^3);

if isempty(lambda_res) || g2 == 0
    % fallback if no residual eigenvalues (unlikely) or numerical issues
    Q_lim = prctile(SPE_train, 95);
else
    h0 = 1 - (2*g1*g3) / (3*g2^2);
    z_alpha = norminv(alpha);
    if h0 <= 0
        Q_lim = prctile(SPE_train, 95);
    else
        Q_lim = g1 * ( (z_alpha * sqrt(2*g2*h0^2) / g1) + 1 + (g2*h0*(h0-1) / g1^2) )^(1/h0);
    end
end

%% ------------------------
%% Fault indices and first-detection sample
T2_fault_idx  = find(T2_39 > T2_lim);
SPEx_fault_idx = find(SPEx > Q_lim);

if ~isempty(T2_fault_idx),  T2_fault_start  = min(T2_fault_idx);  else T2_fault_start  = NaN; end
if ~isempty(SPEx_fault_idx), SPEx_fault_start = min(SPEx_fault_idx); else SPEx_fault_start = NaN; end

fprintf('First T^2 fault index (WT39): %s\n', num2str(T2_fault_start));
fprintf('First SPE fault index (WT39): %s\n', num2str(SPEx_fault_start));

%% ------------------------
%% Visualize T^2 and SPE for WT39
fig3 = figure('Position',[200 200 800 600]);
subplot(2,1,1)
plot(T2_39,'b-','LineWidth',1.2); hold on;
yline(T2_lim,'r--','LineWidth',1.5);
title('Hotelling''s T^2 (WT39)'); xlabel('Sample'); ylabel('T^2'); grid on;

subplot(2,1,2)
plot(SPEx,'b-','LineWidth',1.2); hold on;
yline(Q_lim,'r--','LineWidth',1.5);
title('SPE / Q-statistic (WT39)'); xlabel('Sample'); ylabel('SPE'); grid on;
sgtitle('PCA Monitoring Metrics for WT39')
saveas(fig3,'pca_metrics_wt39.png');

%% ------------------------
%% Reconstruction-error based validation (balanced validation set)
% Use WT_train_scaled training reconstruction errors (SPE_train) for threshold
threshold_recon = prctile(SPE_train, 95);

% Compute reconstruction on validation set using training W
PC_val = WT_val_scaled * W(:,1:pcs);
WT_val_recon = PC_val * W(:,1:pcs)';
val_error = mean((WT_val_scaled - WT_val_recon).^2, 2);   % mean squared error per sample

pred_labels_recon = double(val_error > threshold_recon);
accuracy_recon = mean(pred_labels_recon == labels_val);
fprintf('Validation accuracy (reconstruction threshold): %.2f%%\n', accuracy_recon*100);

confMat_recon = confusionmat(labels_val, pred_labels_recon);
disp('Confusion matrix (reconstruction detector) [Healthy; Faulty]:'), disp(confMat_recon);

% Plot reconstruction error distribution
fig4 = figure;
histogram(SPE_train, 50, 'FaceColor','g'); hold on;
histogram(val_error(labels_val==1), 50, 'FaceColor','r');
xline(threshold_recon,'k--','LineWidth',1.5);
legend('SPE (Healthy Training)','Reconstruction Error (Faulty Validation)','Threshold');
xlabel('Error'); ylabel('Frequency'); title('Reconstruction Error Distribution');
grid on; saveas(fig4,'recon_error_hist.png');

%% ------------------------
%% PCA score projection plots (training vs validation)
PC_train_plot = PC_train;
PC_val_plot = PC_val;

fig5 = figure('Position',[100 100 1200 600]);
subplot(1,2,1)
plot(PC_train_plot(:,1), PC_train_plot(:,2), 'g.'); hold on;
plot(PC_val_plot(labels_val==0,1), PC_val_plot(labels_val==0,2), 'bo');
plot(PC_val_plot(labels_val==1,1), PC_val_plot(labels_val==1,2), 'r+');
xlabel('PC1'); ylabel('PC2'); title('2D PCA Projection'); legend('Healthy Train','Healthy Val','Faulty Val'); grid on;

subplot(1,2,2)
plot3(PC_train_plot(:,1), PC_train_plot(:,2), PC_train_plot(:,3), 'g.'); hold on;
plot3(PC_val_plot(labels_val==0,1), PC_val_plot(labels_val==0,2), PC_val_plot(labels_val==0,3), 'bo');
plot3(PC_val_plot(labels_val==1,1), PC_val_plot(labels_val==1,2), PC_val_plot(labels_val==1,3), 'r+');
xlabel('PC1'); ylabel('PC2'); zlabel('PC3'); title('3D PCA Projection'); grid on;
sgtitle('PCA: Training vs Validation Projection'); saveas(fig5,'pca_proj_train_val.png');

%% ------------------------
%% Loadings plot (interpretation)
fig6 = figure;
bar(W(:,1:3));
xlabel('Sensor Index'); ylabel('Loading'); legend('PC1','PC2','PC3'); title('Sensor Loadings (PC1-PC3)'); grid on;
saveas(fig6,'loadings_pc123.png');

%% ------------------------
%% PLS-DA (supervised classification)
% Use the scaled combined data for training/testing (WT_train_all_scaled, WT_test_all_scaled)
ncomp_pls = min(10, floor(size(WT_train_all_scaled,1)/2)); % reasonable upper bound
ncomp_pls = min(ncomp_pls, pcs); % don't exceed pcs for interpretability

% Train PLS regression (Y is binary)
[XL, YL, XS, YS, BETA, PCTVAR, MSE, stats] = plsregress(WT_train_all_scaled, labels_train, ncomp_pls);

% Predict on test set
WT_test_aug = [ones(size(WT_test_all_scaled,1),1) WT_test_all_scaled];
Ypred_test = WT_test_aug * BETA;          % continuous predicted score
Yclass_test = double(Ypred_test >= 0.5);   % threshold 0.5 -> class

% Evaluate against labels_val
accuracy_pls = mean(Yclass_test == labels_val);
fprintf('Validation Accuracy for PLS-DA: %.2f%%\n', accuracy_pls*100);
confMat_pls = confusionmat(labels_val, Yclass_test);
disp('Confusion matrix (PLS-DA) [Healthy; Faulty]:'), disp(confMat_pls);

% Plot predicted scores (test)
fig7 = figure;
plot(Ypred_test, '-o'); hold on;
yline(0.5,'r--','Threshold'); xlabel('Test sample index'); ylabel('Predicted score');
title('PLS-DA predicted scores (Test set)'); grid on; saveas(fig7,'plsda_scores.png');

% VIP-like insight: variable weights (use weights from 'stats' if available)
if isfield(stats,'W')
    % compute variable importance approximation (sum of abs of weights across components)
    varImp = sum(abs(stats.W(:,1:ncomp_pls)),2);
    fig8 = figure; bar(varImp); xlabel('Sensor'); ylabel('Importance'); title('Approx. Variable Importance (PLS weights)'); grid on; saveas(fig8,'pls_varimp.png');
end

%% ------------------------
%% Summary prints
fprintf('\nSummary:\n');
fprintf('- Trained PCA on WT2 (n_train = %d samples, %d variables).\n', n_train, size(WT_train_scaled,2));
fprintf('- PCA pcs used: %d\n', pcs);
fprintf('- T^2 limit (alpha=%.2f): %.4g\n', alpha, T2_lim);
fprintf('- SPE (Q) limit (alpha=%.2f): %.4g\n', alpha, Q_lim);
fprintf('- First SPE fault index (WT39): %s\n', num2str(SPEx_fault_start));
fprintf('- First T^2 fault index (WT39): %s\n', num2str(T2_fault_start));
fprintf('- Reconstruction-based validation accuracy: %.2f%%\n', accuracy_recon*100);
fprintf('- PLS-DA validation accuracy: %.2f%%\n', accuracy_pls*100);

%% End of script
