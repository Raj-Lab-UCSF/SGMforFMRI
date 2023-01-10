function [pred] = predict_SGMforfMRI(data,hyperparams)
%predict_SGMforfMRI Uses the SGM model to predict FC and BOLD PSD from SC
%   Works with SGM_for_fMRI_script to predict FC and BOLD PSD from input
%   SC. 
%   Inputs:
%       data: struct with fields SC, FC, Fx, and eig_weights
%       hyperparams: struct of hyperparameters with fields Pwstr, params0,...
%           ll, ul, maxiter, eig_start, eigfrac, omega0
%   Outputs:
%       pred: structure of predictions and their maximum correlation


% Load Hyperparameters:
Pwstr = hyperparams.Pwstr; % options: 'ones' 'randn' 'none' 'U' 'Diag' % Specifies which input driving function to use
params0 = hyperparams.params0;  %initalization for [alpha, tau]
ll = hyperparams.ll; %lower bound for [alpha; tau]
ul = hyperparams.ul; %upper bound for [alpha; tau]
maxiter = hyperparams.maxiter; %maximum itterations for optimization function
errfact = 1; % weighs SCFC vs SGM in the error function to be minimized


if isequal(hyperparams.FC_omega, 'MaxAmp')
    %Define angular frequency at which to compute predicted FC based on
    %mean frequency at which Fx achieves max power:
    [~, max_idx] = max(data.Fx);
    f_at_max = data.fvec(max_idx);
    omega = 2*pi*mean(f_at_max);
elseif isfloat(hyperparams.FC_omega)
    omega = 2*pi*mean(hyperparams.FC_omega);
elseif isequal(hyperparams.FC_omega, 'FxMean')
    omega = [];
else
    disp("Please enter either a frequency or 'MaxAmp' or 'FxMean' for hyperparams.FC_omega")
    return
end

omegavec = 2*pi*data.fvec(:);  % nt x 1 vector

nt = length(omegavec);    
nzinds = find(triu(data.FC,1));
pred = struct('params_opt', [], 'FC', [], 'Fx', [], 'minerr', [], 'Rmax_SCFC', [], 'Rmax_SGM', []);


% Compute Laplacian and its eigens
SC = data.SC;
rowdegree = (sum(SC, 2)).';
coldegree = sum(SC, 1);
nroi_ = length(rowdegree);

L = eye(nroi_) - diag(1./(sqrt(rowdegree)+eps)) * SC* diag(1./(sqrt(coldegree)+eps)) ;

[U, ev] = eig(L);
ev = diag(ev);
[~, ii] = sort(ev, 'ascend');
ev = ev(ii);
U = U(:,ii);

ev_weight = data.eig_weights;


% Optimization using fmincon:
params_opt_ = fmincon(@myfun_both, params0, [], [], [], [], ll, ul, [], optimoptions('fmincon','Display','none','MaxIter', maxiter));
[err_both_, r_SCFC_, r_SGM_] = myfun_both(params_opt_);
pred.params_opt = params_opt_;
FCpred_ = ForwardModel_SCFC(params_opt_);
pred.FC = FCpred_;
pred.minerr = err_both_;
pred.Rmax_SCFC = r_SCFC_;
Fxpred_ = ForwardModel_SGM(params_opt_);
pred.Fx = Fxpred_;
pred.Rmax_SGM = r_SGM_;

%% Internal Functions for optimization:

function [err_both, r_SCFC, r_SGM] = myfun_both(params)
    [err_SCFC, r_SCFC] = myfun_SCFC(params);
    [errvec, rvec] = myfun_SGM(params);
    r_SGM = nanmean(rvec);
    err_SGM = nanmean(errvec);

    err_both = err_SCFC + errfact*err_SGM;
end

% FUNCTIONAL CONNECTIVITY SGM PREDICTION:
function [err, r] = myfun_SCFC(params)
    outFC = ForwardModel_SCFC(params);

    r = corr(data.FC(nzinds), outFC(nzinds));
    err = 1-r; 

end

function outFC = ForwardModel_SCFC(params)
% This is the forward SCFC eigen model, that gives predicted FC from SC, and
% returns the best Pearson R between model and empirical FC
    alpha = params(1); 
    tau = params(2);     
    if length(params)>2
        tau_f = params(3);
    else 
        tau_f = tau;
    end
    
    if isequal(hyperparams.FC_omega, 'FxMean')
        omegavec_ = omegavec; 
    else
        omegavec_ = omega; 
    end

    tmp = zeros(length(ev),1);

%     for j = 1:length(omegavec_)
%         omega = omegavec_(j);
%         He = 1/tau_f^2./(1i*omega+1/tau_f).^2;
%         newev = 1 ./ (1i*omega + 1/tau*He*(1- alpha*(1 - ev))); %new line
%         newev = (abs(newev)).^2;
%         tmp = tmp + newev;
%     end
% 
%     newev = tmp/length(omegavec_).*ev_weight';
%     outFC = U *  bsxfun(@times, newev(:), U');
%     dg = 1./(1e-4+sqrt(diag(outFC)));
%     outFC = bsxfun(@times, outFC, dg);
%     outFC = bsxfun(@times, outFC, dg.');

% DEBUG, remove later
    outFC = zeros(nroi_);
    for j = 1:length(omegavec_)
        omega = omegavec_(j);
        He = 1/tau_f^2./(1i*omega+1/tau_f).^2;
        newev = 1 ./ (1i*omega + 1/tau*He*(1- alpha*(1 - ev))); %new line
        newev = (abs(newev)).^2 .*ev_weight';
        tmp = U *  bsxfun(@times, newev(:), U');
        dg = 1./(1e-4+sqrt(diag(tmp)));
        tmp = bsxfun(@times, tmp, dg);
        tmp = bsxfun(@times, tmp, dg.');
        outFC = outFC + tmp;
    end
    outFC = outFC/length(omegavec_);
end  




% BOLD PSD SGM PREDICTION:
function [errvec, rvec] = myfun_SGM(params)
    outFx = ForwardModel_SGM(params);  % evaluate model at params
    rvec = nan(nroi_,1);

    for n = 1:nroi_
        qdata = abs(data.Fx(:, n));
     
        qmodel = abs(outFx(:,n));

        rvec(n) = corr(qdata, qmodel, 'type', 'Pearson', 'rows', 'complete'); 

    end

    rvec(isnan(rvec)) = 0;
    errvec = abs(1 - rvec);

end



function outFx = ForwardModel_SGM(params)
    % This is the forward model, that gives predicted fMRI freq spectrum from SC
    % Output: matrix of size nt x nroi (in Fourier, not time)
    
    alpha = params(1); 
    tau = params(2);     
    if length(params)>2
        tau_f = params(3);
    else 
        tau_f = tau;
    end

    He = 1/tau_f^2./(1i*omegavec.'+1/tau_f).^2;
    spectral_component = 1 ./ bsxfun(@plus, 1i*omegavec.', 1/tau*(1 - alpha*(1 - ev)) *He );  % k x nt % new line
    
    spectral_component_weighted = spectral_component.*ev_weight';
    
    % Define UtP (driving function) with varying size: k x 1 (for ones, randn) or 1 (for U) or k x nroi (for none)
    switch Pwstr
        case 'ones',  UtP = U' * ones(nroi_,1);
        case 'randn', UtP = U' * randn(nroi_,1);
        case 'U', UtP = 1;
        case 'none', UtP = U';
    end
    outFx = zeros(nt, nroi_);
    
    
    for n = 1:nt
    
        if strcmp(Pwstr, 'Diag')
            tmp =  U * bsxfun(@times, spectral_component_weighted(:,n), U');
            outFx(n,:) = (vecnorm(tmp, 2, 2)).'; % 1 x nroi
        else
            tmp =  U * bsxfun(@times, spectral_component_weighted(:,n), UtP); %new line

            if strcmp(Pwstr, 'none')
                outFx(n,:) = (vecnorm(tmp, 2, 2)).'; % 1 x nroi
            else
                outFx(n,:) = tmp.';  % 1 x nroi
            end

        end

    end




end



% END OF PREDICTION FUNCTION



end