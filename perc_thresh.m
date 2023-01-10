function [A_thresh,thresh] = perc_thresh(A)
%PERC_THRESH Generates a matrix at the percolation threshold of A.
%   The percolation threshold is the minimal fully connected graph of
%   matrix A. All weakest links below the threshold are removed. Removing
%   the link at the threshold will result in a disconnected graph.
%
%   This function returns a thresholded matrix (A_thresh) and optionally
%   returns the threshold weight (thresh) used.
%   Written by Benjamin Sipes at UCSF, 2022.
%
%   Note:
%   This function uses the get_components function (below) from the
%   Brain Connectivity Toolbox. All credit for that function goes to it's
%   creators. I have made no modifications to that function.
%   For it's documentation-- https://sites.google.com/site/bctnet/


A(isnan(A)) = 0; %remove all nans
A = double(A);
for i = 1:size(A,1), A(i,i) = 0; end %zero the diagonal

% Get number of components of A

C = max(get_components(A));

A_thresh = A;

while C==1


    %Threshold minimum weight:
    thresh = min(abs(nonzeros(A_thresh)),[],'all');
    A_thresh(abs(A)<=thresh) = 0;

    %Obtain number of components in graph
    C = max(get_components(A_thresh));
%     disp(thresh);

end

%Final thresholding
A_thresh = A;
A_thresh(abs(A)<(thresh+eps)) = 0; %note this is < and not <= as above.

end

function [comps,comp_sizes] = get_components(adj)
% GET_COMPONENTS        Connected components
%
%   [comps,comp_sizes] = get_components(adj);
%
%   Returns the components of an undirected graph specified by the binary and 
%   undirected adjacency matrix adj. Components and their constitutent nodes are 
%   assigned the same index and stored in the vector, comps. The vector, comp_sizes,
%   contains the number of nodes beloning to each component.
%
%   Inputs:         adj,    binary and undirected adjacency matrix
%
%   Outputs:      comps,    vector of component assignments for each node
%            comp_sizes,    vector of component sizes
%
%   Note: disconnected nodes will appear as components of size 1
%
%   J Goni, University of Navarra and Indiana University, 2009/2011

if size(adj,1)~=size(adj,2)
    error('this adjacency matrix is not square');
end

if ~any(adj-triu(adj))
  adj = adj | adj';
end

%if main diagonal of adj do not contain all ones, i.e. autoloops
if sum(diag(adj))~=size(adj,1)
    
    %the main diagonal is set to ones
    adj = adj|speye(size(adj));
end

%Dulmage-Mendelsohn decomposition
[~,p,~,r] = dmperm(adj); 

%p indicates a permutation (along rows and columns)
%r is a vector indicating the component boundaries

% List including the number of nodes of each component. ith entry is r(i+1)-r(i)
comp_sizes = diff(r);

% Number of components found.
num_comps = numel(comp_sizes);

% initialization
comps = zeros(1,size(adj,1)); 

% first position of each component is set to one
comps(r(1:num_comps)) = ones(1,num_comps); 

% cumulative sum produces a label for each component (in a consecutive way)
comps = cumsum(comps); 

%re-order component labels according to adj.
comps(p) = comps; 
end