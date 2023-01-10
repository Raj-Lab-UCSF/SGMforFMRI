function [eig_weights, proj_FC, U, ev, L] = get_eigenmode_weights(SC, TS, FC, method)
%get_eigenmode_weights Returns a vector of SC eigenmode weights based on
%the input time series (TS).
%   Inputs: 
%       SC: Structural Connectivity Matrix (NxN)
%       TS: fMRI Time-Series Matrix (timeXN)
%       method: how to compute weights; options = 'PI', or 'OD', or 'PI_OD'
%           'PI' = Participation Index (time series based)
%           'OD' = Off-Diagonal (FC based)
%           'PI_OD' = the multiplication of PI and OD
%   Outputs: 
%       eig_weights: eigenmode weighting by selected method (Nx1)
%       U: Structural Laplacian eigenvectors (NxN)
%       ev: Structural Laplacian eigenvalues (Nx1)
%       L: Structural Laplacian (NxN)

%Compute Laplacian
rowdegree = (sum(SC, 2)).';
coldegree = sum(SC, 1);
nroi_ = length(rowdegree);
L = eye(nroi_) - diag(1./(sqrt(rowdegree)+eps)) * SC* diag(1./(sqrt(coldegree)+eps)) ;

[U, ev] = eig(L);
ev = diag(ev);
[~, ii] = sort(ev, 'ascend');
ev = ev(ii);
U = U(:,ii);

switch method
    case 'PI'
        % % eigenmode Participation Index (PI) weighting:
        eig_partic_ts = TS*U; %Projection of TS into U space

        
        eig_partic_mean = mean(abs(eig_partic_ts)); %L-1 Norm
         
        % Normalize
        ev_weight_PI = eig_partic_mean./sum(eig_partic_mean(:)); %makes all eigenmodes ratioes of each other
        eig_weights = ev_weight_PI;

    case 'OD'
        % % Off Diagonal (OD) weighting:
        proj_FC = abs(U'*FC*U);  % Meaningful eigenmodes have high on-diagonal and low off-diagonal
        rsum = sum(proj_FC, 2) + eps;
        csum = sum(proj_FC, 1) + eps;
        %try geometric mean of all off-diag entires (excluding the diag)

        % get diag / off-diag ratio: 1 for totally diag, 0 for totally off-diag
        eig_offdiag_wts = diag(proj_FC)./ sqrt(rsum(:).*csum(:)); 
        ev_weight_OD = (eig_offdiag_wts./sum(eig_offdiag_wts))';
        eig_weights = ev_weight_OD;

    case 'PI_OD'
        % % eigenmode Participation Index (PI) weighting:
        TS = smoothdata(TS,'gaussian',5); %smooth TS to get better projection estimates
        eig_partic_ts = TS*U;
        eig_partic_mean = mean(abs(eig_partic_ts));
    
        % % Create eigenmode ratios from the mean participation index
        ev_weight_PI = eig_partic_mean./sum(eig_partic_mean(:)); %makes all eigenmodes ratioes of each other

        % % Off Diagonal (OD) weighting:
        proj_FC = abs(U'*FC*U);  % Meaningful eigenmodes have high on-diagonal and low off-diagonal
        rsum = sum(proj_FC, 2) + eps;
        csum = sum(proj_FC, 1) + eps;

        % get diag / off-diag ratio: 1 for totally diag, 0 for totally off-diag
        eig_offdiag_wts = diag(proj_FC)./ sqrt(rsum(:).*csum(:)); 
        ev_weight_OD = (eig_offdiag_wts./sum(eig_offdiag_wts))';

        % % Combine PI and OD:
        ev_weight_PI_OD = ev_weight_PI.*ev_weight_OD;
        eig_weights = ev_weight_PI_OD./sum(ev_weight_PI_OD);

    case 'On_Diag'
        proj_FC = abs(U'*FC*U);  % Meaningful eigenmodes have high on-diagonal of the SC projection
        eig_weights = diag(proj_FC)'; %I was asked not to normalize this.

    case 'ones'
        %Basically no eigen-weighting
        eig_weights = ones(1, nroi_);


end



end