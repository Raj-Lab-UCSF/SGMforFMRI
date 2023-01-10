%% SGM for fMRI Script

% SGM for fMRI: a Spectral Graph Model (SGM) for predicting the brain's
% functional connectivity (FC) and time series power spectral density from 
% its structural connectivity (SC).
% Written by Ashish Raj and Benjamin Sipes, UCSF, July 2022.

%% Part 1: Import data

dataset = 'Mathalon'; % Options: 'Mathalon' or 'MICA'
use_Cadd = true;
wopt0 = 0.5; %Optimal weighting for Cadj and Cinterhem

switch dataset
    case 'Mathalon'

        RAW_data = load('Mathalon_data_struct.mat');
        RAW_data = RAW_data.SGM_example_data;

        %Dataset Specific fMRI parameters
        TR = 2; %fMRI TR in seconds
        Fs = 1/TR; %Sampling Frequency
        nfft = 64; %number of frequency points to estimate
        fmax = 0.15; %max for Mathalon dataset is 0.25
        pwelch_windows = 6;

        if use_Cadd
            Cadj = load('adjacency_natural_order.mat');
            Cadj = Cadj.adjacent_regions; 
            Cadj = Cadj/sum(sum(Cadj));
            Cinterhem = load('interhem_mat_natural_order.mat'); 
            Cinterhem = Cinterhem.interhem_mat; 
            Cinterhem = Cinterhem/sum(sum(Cinterhem));

            C_add = wopt0*Cadj + wopt0*Cinterhem;
        end


    case 'MICA'
        RAW_data = load('MICA_aparc_struct.mat');
        RAW_data = RAW_data.MICA;
        nsubjects = length(RAW_data);

        TR = 0.6; %fMRI TR in seconds
        Fs = 1/TR; %Sampling Frequency
        nfft = 256; %number of frequency points to estimate
        fmax = 0.25; %max for MICA dataset is 0.83
        pwelch_windows = 10;

        if use_Cadd

            collosum_idx = [14+4, 14+35+4]; %Corpus Collosum is the 4th idx of cerebrum (first 14 idx are subcortex and there's 35 cerebral nodes per hemisphere)
            
            for i = 1:nsubjects
            
                SC_tmp = RAW_data(i).SC;
                SC_tmp(:,collosum_idx) = [];
                SC_tmp(collosum_idx,:) = [];
            
                RAW_data(i).SC = SC_tmp;
            
                TS_tmp = RAW_data(i).fMRI_TS;
                TS_tmp(:,collosum_idx) = [];
        
                [TS_pca, TS_scores] = pca(TS_tmp, 'Economy',false);
                TS_recon = TS_scores(:,2:end)*TS_pca(:,2:end)'; %remove first PC
            
                RAW_data(i).fMRI_TS = TS_recon;
            
            end
            clear SC_tmp TS_tmp TS_recon TS_scores;
        
            % % Interhemisphere connections for MICA 82x82:
            Cinterhem = load('interhem.mat'); 
            Cinterhem = Cinterhem.interhem; 
            Cinterhem = Cinterhem/sum(sum(Cinterhem));

            C_add = wopt0*Cinterhem;
            % 
            
        end

end


%% Part 2: Prepare data

%Model Decisions:
perc_thresh_method = 'Individual'; %Options: 'Individual', 'Mean', 'None'
ev_weight_method = 'On_Diag'; %options: 'PI', 'OD', 'PI_OD', 'On_Diag', or 'ones' (for no weighting)
group_av_ev_weight = true; %determines whether to reset all weights to be same across subjects (the mean)
eigstart = 2; 

%Get number of subjects in data:
nsubjects = length(RAW_data);

%Get number of rois x number of time points:
[nt, nroi] = size(RAW_data(1).fMRI_TS);

eigkeep = zeros(1,nroi); eigkeep(eigstart:end) = 1;

%Note: fMRI frequency parameters defined with dataset loading above
% frequency vector (fvec):
fvec = linspace(0.01, fmax, nfft);

%number of edges:
combos = nchoosek(1:nroi,2);

data_empirical = struct('SC', [], 'FC', [], 'Fx', [], 'fvec', [], 'eig_weights', []);

% %Find mean percolation threshold, if desired:
if strcmp(perc_thresh_method, 'Mean')
    thresh_perc_subs = zeros(nsubjects,1);
    for i = 1:nsubjects
    
        fmri_ts = RAW_data(i).fMRI_TS;
        fx1 = bsxfun(@minus, fmri_ts, mean(fmri_ts,1)); %demean
        fx1 = detrend(fx1); %detrend
        fx1 = lowpass(fx1,fmax,Fs);
    
        Pearson_FC = corr(fx1); %Pearson's correlation matrix
    
    %     for j = 1:length(combos), Pearson_FC(combos(j,1), combos(j,2)) = Pearson_FC(combos(j,2), combos(j,1)); end %force symmetry
    %     data_empirical(i).FC = perc_thresh(Pearson_FC);
        [~, thresh_perc_subs(i)] = perc_thresh(Pearson_FC);
    
    
    end
    mean_perc_thresh = mean(thresh_perc_subs);
end

%Main Data preparation stage:

for i = 1:nsubjects

    %Structural Connectivity:
    
    SC_sub = RAW_data(i).SC; %Raw diffusion connectivity matrix (SC)
    SC_norm = SC_sub/sum(SC_sub(:)); %noramlize SC by total network weight

    if use_Cadd
        SC_norm_Cadd = SC_norm+C_add;
        data_empirical(i).SC = SC_norm_Cadd/sum(SC_norm_Cadd(:)); %renormalize and save
    else
        data_empirical(i).SC = SC_norm;
    end

    

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    %Functional Connectivity:
    fmri_ts = RAW_data(i).fMRI_TS;
    ts_proc = bsxfun(@minus, fmri_ts, mean(fmri_ts,1)); %demean
    ts_proc = detrend(ts_proc); %detrend
    ts_proc = lowpass(ts_proc,fmax,Fs); %remove all frequencies above what we are estimating

    Pearson_FC = corr(ts_proc); %Pearson's correlation matrix

    switch perc_thresh_method
        case 'Mean'
            Pearson_FC(abs(Pearson_FC)<mean_perc_thresh) = 0; %threshold based on mean percolation threshold
        case 'Individual'
            Pearson_FC = perc_thresh(Pearson_FC);
    end

% % % Individualized, above...
    for j = 1:length(combos), Pearson_FC(combos(j,1), combos(j,2)) = Pearson_FC(combos(j,2), combos(j,1)); end %force symmetry
    data_empirical(i).FC = Pearson_FC;

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    %BOLD Spectra:
    data_empirical(i).Fx = sqrt( pwelch(ts_proc, round(nt/pwelch_windows), [], fvec, Fs) ); 
    data_empirical(i).fvec = fvec;

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    %Define Eigenmode Weighting
    data_empirical(i).eig_weights = get_eigenmode_weights(data_empirical(i).SC, ts_proc, Pearson_FC, ev_weight_method) .* eigkeep;

end

%Create uniform eigenvalue weighting, if desired:
if group_av_ev_weight

    eig_weights_all = zeros(nsubjects,nroi);

    for i = 1:nsubjects
        eig_weights_all(i,:) = [data_empirical(i).eig_weights];
    end

    ev_uniform = mean(eig_weights_all,1);

    for i = 1:nsubjects
        data_empirical(i).eig_weights = ev_uniform;
    end

end


%% Generate Predictions
tic;
SGM_METHOD = 'Normal'; %options: "Normal"; "Diag"; "TWOTAU"

% Define Hyperparameters for SGM:
switch SGM_METHOD
    case 'Normal'
        hyperparams.Pwstr = 'ones'; % options: 'ones' 'randn' 'none' 'U' % Specifies which input driving function to use
        hyperparams.params0 = [1; 2];  %initalization for [alpha, tau]
        hyperparams.ll = [0 ; 0]; %lower bound for [alpha; tau]
        hyperparams.ul = [1 ; 5]; %upper bound for [alpha; tau]
%         hyperparams.ul = [5 ; 5]; %upper bound for [alpha; tau]

    case 'TWOTAU'
        hyperparams.Pwstr = 'ones'; % options: 'ones' 'randn' 'none' 'U' % Specifies which input driving function to use
        hyperparams.params0 = [0.5; 2; 2];  %initalization for [alpha, tau, tau_f]
        hyperparams.ll = [0; 0; 0]; %lower bound for [alpha; tau; tau_f]
        hyperparams.ul = [1; 10; 10]; %upper bound for [alpha; tau; tau_f]

    case 'Diag'
        hyperparams.Pwstr = 'Diag'; % options: 'ones' 'randn' 'none' 'U' % Specifies which input driving function to use
        hyperparams.params0 = [1; 2];  %initalization for [alpha, tau]
        hyperparams.ll = [0 ; 0]; %lower bound for [alpha; tau]
        hyperparams.ul = [5 ; 5]; %upper bound for [alpha; tau]

end

hyperparams.maxiter=2000; %maximum itterations for optimization function
hyperparams.FC_omega = 'FxMean'; % either input specific frequency or 'MaxAmp'(uses mean frequency with maximum amplitude for each individual subject), or 'FxMean' to use all frequencies

%Main Prediction Loop:
pred = struct('params_opt', [], 'FC', [], 'Fx', [], 'minerr', [], 'Rmax_SCFC', [], 'Rmax_SGM', []);


for i = 1:nsubjects
    
    pred(i) = predict_SGMforfMRI(data_empirical(i), hyperparams);

end


% % Display Mean R results:
fprintf('Time To Compute: %0.2f minutes \n',toc/60);
fprintf('Mean(STD) R SCFC: %0.2f (%0.2f) \n', mean([pred(:).Rmax_SCFC]), std([pred(:).Rmax_SCFC]));
fprintf('Mean(STD) R SGM: %0.2f (%0.2f) \n', mean([pred(:).Rmax_SGM]), std([pred(:).Rmax_SGM]));
params_opt = [pred(:).params_opt]';
fprintf('Mean(STD) Alpha: %0.2f (%0.2f) \n', mean(params_opt(:,1)), std(params_opt(:,1)));
fprintf('Mean(STD) Tau: %0.2f (%0.2f) \n', mean(params_opt(:,2)), std(params_opt(:,2)));

% save('Workspace_FULL_SGMforFMRI_Mathalon.mat');
% save('Workspace_FULL_SGMforFMRI_MICA.mat');
%% Visualize Results:

%subject list:
s = [22, 30, 49];
% s = [21, 37, 41];
% s = [21, 37, 48];



params_opt = [pred(:).params_opt]';

figure;
Spec_ha = tight_subplot(3,2,[0.05, 0.05]);

axes(Spec_ha(1)); plot(fvec, abs(data_empirical(s(1)).Fx)); title('Empirical fMRI spectra'); ylabel({sprintf('Subject %d',s(1));'Magnitude'}); 
fact = ( (pred(s(1)).Fx(:))' * data_empirical(s(1)).Fx(:) )/ norm(pred(s(1)).Fx(:))^2;
axes(Spec_ha(2)); plot(fvec, abs(fact*pred(s(1)).Fx)); title('Predicted spectra'); annotation('textbox',[0.85,0.83,0.1,0.1],'String',{sprintf('R = %0.2f', pred(s(1)).Rmax_SGM), sprintf('alpha = %0.2f', params_opt(s(1),1)), sprintf('tau = %0.2f', params_opt(s(1),2))});
axes(Spec_ha(3)); plot(fvec, abs(data_empirical(s(2)).Fx)); ylabel({sprintf('Subject %d',s(2));'Magnitude'});
fact = ( (pred(s(2)).Fx(:))' * data_empirical(s(2)).Fx(:) )/ norm(pred(s(2)).Fx(:))^2;
axes(Spec_ha(4)); plot(fvec, abs(fact*pred(s(2)).Fx)); annotation('textbox',[0.85,0.52,0.1,0.1],'String',{sprintf('R = %0.2f', pred(s(2)).Rmax_SGM), sprintf('alpha = %0.2f', params_opt(s(2),1)), sprintf('tau = %0.2f', params_opt(s(2),2))});
axes(Spec_ha(5)); plot(fvec, abs(data_empirical(s(3)).Fx));xlabel('Frequency (Hz)'); ylabel({sprintf('Subject %d',s(3));'Magnitude'});
fact = ( (pred(s(3)).Fx(:))' * data_empirical(s(3)).Fx(:) )/ norm(pred(s(3)).Fx(:))^2;
axes(Spec_ha(6)); plot(fvec, abs(fact*pred(s(3)).Fx)); xlabel('Frequency (Hz)'); annotation('textbox',[0.85,0.2,0.1,0.1],'String',{sprintf('R = %0.2f', pred(s(3)).Rmax_SGM), sprintf('alpha = %0.2f', params_opt(s(3),1)), sprintf('tau = %0.2f', params_opt(s(3),2))});



FC_FIG = figure;
cnt = length(s);
for z = 1:cnt
    subplot(cnt,3,3*(z-1)+1); imagesc(log(data_empirical(s(z)).SC)); axis square; title('SC (log scale)'); set(gca, 'YTickLabel',[]); set(gca, 'XTickLabel',[]); 
    subplot(cnt,3,3*(z-1)+2); imagesc(data_empirical(s(z)).FC); axis square; title('Empirical FC'); set(gca, 'YTickLabel',[]); set(gca, 'XTickLabel',[]); 
    subplot(cnt,3,3*(z-1)+3); imagesc(pred(s(z)).FC); axis square; title(sprintf('Predicted FC, R = %0.2f', pred(s(z)).Rmax_SCFC)); set(gca, 'YTickLabel',[]); set(gca, 'XTickLabel',[]); 
end


Hist_FIG = figure;
subplot(2,2,1);
% histogram([pred(:).Rmax_SCFC],0.35:0.05:0.95,'FaceColor','#0072BD'); 
histogram([pred(:).Rmax_SCFC],0:0.05:0.95,'FaceColor','#0072BD'); 
title('Predicted Functional Connectivity R','FontSize',10);
% xline(mean([pred(:).Rmax_SCFC]), '--k', ['mean R = ' num2str(mean([pred(:).Rmax_SCFC]))], 'LineWidth',2);
xline(mean([pred(:).Rmax_SCFC]), '--k', 'LineWidth',2);
% xlim([0.35 0.95]); 
ylim([0 20]);

subplot(2,2,2);
% histogram([pred(:).Rmax_SGM], 0.35:0.05:0.95,'FaceColor','#77AC30'); 
histogram([pred(:).Rmax_SGM], 0:0.05:0.95,'FaceColor','#77AC30'); 
title('Predicted fMRI Spectra R','FontSize',10);
% xline(mean([pred(:).Rmax_SGM]), '--k', ['mean R = ' num2str(mean([pred(:).Rmax_SGM]))], 'LineWidth',2);
xline(mean([pred(:).Rmax_SGM]), '--k', 'LineWidth',2);
% xlim([0.35 0.95]); 
ylim([0 20]);

subplot(2,2,3);
% histogram(params_opt(:,1),0.6:0.05:1,'FaceColor','#EDB120'); 
histogram(params_opt(:,1),0:0.05:1,'FaceColor','#EDB120'); 
title('$\alpha$','interpreter','latex','FontWeight','bold','FontSize',20);
% xline(mean(params_opt(:,1)), '--k', ['mean \alpha = ' num2str(mean(params_opt(:,1)))], 'LineWidth',2);
xline(mean(params_opt(:,1)), '--k', 'LineWidth',2);
ylim([0 20]);

subplot(2,2,4);
% histogram(params_opt(:,2),0.5:0.25:5,'FaceColor','#7E2F8E'); 
histogram(params_opt(:,2),0:0.25:5,'FaceColor','#7E2F8E'); 
title('$\tau$','interpreter','latex','FontWeight','bold','FontSize',20);
% xline(mean(params_opt(:,2)), '--k', ['mean \tau = ' num2str(mean(params_opt(:,2)))], 'LineWidth',2);
xline(mean(params_opt(:,2)), '--k', 'LineWidth',2);
ylim([0 20]);

%% Null Models:

%Note, please run the above sections before this section.

data_random = struct('SC', [], 'FC', [], 'Fx', [], 'fvec', [], 'eig_weights', []);
pred_random = struct('params_opt', [], 'FC', [], 'Fx', [], 'minerr', [], 'Rmax_SCFC', [], 'Rmax_SGM', []);


rng('shuffle','simdTwister');

null_model = 'FC'; %options: 'FC' or 'SC'

nrandomizations = 1000;



for i = 1:nrandomizations

    sub=randi(nsubjects);

    
        %Structural Connectivity:
    SC_sub = RAW_data(sub).SC; %Raw diffusion connectivity matrix (SC)
    SC_norm = SC_sub/sum(SC_sub(:)); %noramlize SC by total network weight

    if use_Cadd
        SC_norm_Cadd = SC_norm+C_add;
        SC = SC_norm_Cadd/sum(SC_norm_Cadd(:));
    end
    
    if strcmp(null_model,'SC')
        data_random(i).SC = randmio_und_connected(SC,10);
    else
        data_random(i).SC = SC; %renormalize and save
    end

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    %Functional Connectivity:
    fmri_ts = RAW_data(sub).fMRI_TS;
    ts_proc = bsxfun(@minus, fmri_ts, mean(fmri_ts,1)); %demean
    ts_proc = detrend(ts_proc); %detrend
    ts_proc = lowpass(ts_proc,fmax,Fs); %remove all frequencies above what we are estimating

    if strcmp(null_model,'FC')
        ts_proc = ts_proc(:,randperm(nroi));
    end

    Pearson_FC = corr(ts_proc); %Pearson's correlation matrix

    if strcmp(perc_thresh_method,'Mean')
        Pearson_FC(abs(Pearson_FC)<mean_perc_thresh) = 0; %threshold based on mean percolation threshold
    elseif strcmp(perc_thresh_method,'Individual')
        Pearson_FC = perc_thresh(Pearson_FC);
    end

    for j = 1:length(combos), Pearson_FC(combos(j,1), combos(j,2)) = Pearson_FC(combos(j,2), combos(j,1)); end %force symmetry
    data_random(i).FC = Pearson_FC;

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    %BOLD Spectra:
    data_random(i).Fx = sqrt( pwelch(ts_proc, round(nt/pwelch_windows), [], fvec, Fs) ); 
    data_random(i).fvec = fvec;

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    %Define Eigenmode Weighting
    if group_av_ev_weight && strcmp(null_model,'SC')
        data_random(i).eig_weights = ev_uniform;
        pred_random(i) = predict_SGMforfMRI(data_random(i), hyperparams);
    else
        data_random(i).eig_weights = get_eigenmode_weights(data_random(i).SC, ts_proc, Pearson_FC, ev_weight_method);
    end

    

end


if group_av_ev_weight && strcmp(null_model,'FC')


    eig_weights_rand = zeros(nrandomizations,nroi);

    for i = 1:nrandomizations
        eig_weights_rand(i,:) = [data_random(i).eig_weights];
    end

    ev_uniform = mean(eig_weights_rand,1);

    for i = 1:nrandomizations
        data_random(i).eig_weights = ev_uniform;
    end

    for i = 1:nrandomizations
    
        pred_random(i) = predict_SGMforfMRI(data_random(i), hyperparams);
    
    end
end


save('Null_Model_FC_data_pred.mat','data_random','pred_random');


%% Alternate Models:

NDM_Rmax = zeros(nsubjects,1);
eigen_Rmax = zeros(nsubjects,1);

NDM_FC_pred = zeros(nroi,nroi,nsubjects);
eigen_FC_pred = zeros(nroi,nroi,nsubjects);

for i = 1:nsubjects

    SC_sub = data_empirical(i).SC;
    FC_sub = data_empirical(i).FC;

   [~,NDM_FC_pred(:,:,i),~, NDM_Rmax(i)]  = fit_SCFC_model(SC_sub,FC_sub,'NDM');
   [~,eigen_FC_pred(:,:,i),~, eigen_Rmax(i)]  = fit_SCFC_model(SC_sub,FC_sub,'eigen');

end


save('NDM_eigen_fits_Mathalon.mat','NDM_Rmax','NDM_FC_pred','eigen_Rmax', 'eigen_FC_pred');


%% Compute Geodesic distances

%Get empirical FC for all and take average
for i = 1:nsubjects
    FC_sub = [data_empirical(i).FC];
    Empirical_FC_all(:,:,i) = FC_sub;
    SGM_FC_pred(:,:,i) = [pred_empirical(i).FC].*~eye(nroi);
end
Empirical_FC_mean = mean(Empirical_FC_all,3).*~eye(nroi);

EmpiricalMean_geo = zeros(nsubjects,1);
SGM_geo = zeros(nsubjects,1);
NDM_geo = zeros(nsubjects,1);
eigen_geo = zeros(nsubjects,1);

for i = 1:nsubjects


    EmpiricalMean_geo(i) = geodesic_dist(Empirical_FC_all(:,:,i), Empirical_FC_mean);
    SGM_geo(i) = geodesic_dist(Empirical_FC_all(:,:,i), SGM_FC_pred(:,:,i));
    NDM_geo(i) = geodesic_dist(Empirical_FC_all(:,:,i), NDM_FC_pred(:,:,i));
    eigen_geo(i) = geodesic_dist(Empirical_FC_all(:,:,i), eigen_FC_pred(:,:,i));

end


save('Geodesic_distances_NDM_Eigen_SGM_Mathalon.mat','EmpiricalMean_geo','SGM_geo','NDM_geo','eigen_geo');


    
    