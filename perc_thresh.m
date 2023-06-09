function [A_thresh, thresh] = perc_thresh(A)
% perc_thresh Generates a matrix at the percolation threshold of A..
%   The percolation threshold is the minimal fully connected graph of
%   matrix A. All weakest links below the threshold are removed. Removing
%   the link at the threshold will result in a disconnected graph.
%
%   This function returns a thresholded matrix (A_thresh) and optionally
%   returns the threshold weight (thresh) used.
%   Written by Benjamin Sipes at UCSF, 2022; updated 2023.
% 
%   Update: Now implements an algorithm with a base-2 search for the
%   percolation threshold, resulting in ~6000X speeed-up. Niceee
%
%   Note:
%   This function uses the get_components function (below) from the
%   Brain Connectivity Toolbox. All credit for that function goes to it's
%   creators. I have made no modifications to that function.
%   For it's documentation-- https://sites.google.com/site/bctnet/
% 
%   Usage: [A_thresh, thresh] = perc_thresh(A)


    % Prepare Matrix:
    A(isnan(A)) = 0;
    A = double(A);
    n = size(A,1);
    A = A.*~eye(n); % Set diagonal elements to zero
    A_orig = A;
    degrees = sum(A);
    A(:,find(degrees==0)) = [];
    A(find(degrees==0),:) = [];
    A = double(A);

    
    A_edges = sort(abs(nonzeros(triu(A,1))));

    n_edges = length(A_edges);
    next_power_of_2 = 2^ceil(log2(abs(n_edges)));
    
    A_edges = cat(1, zeros((next_power_of_2 - n_edges),1), A_edges);
    

    while length(A_edges)>1

        % Update the threshold:
        thresh = A_edges(length(A_edges)/2);

        A_thresh = A .* (abs(A) > thresh); % Vectorized thresholding

        % Obtain number of components in graph
        C = max(get_components(A_thresh));

        if C > 1 %the graph is disconnected...
            % Get rid of all stronger edges than the threshold
            A_edges( A_edges > thresh ) = [];

        else %the graph is still connected...
            % Get rid of all weaker edges than the threshold
            A_edges( A_edges <= thresh ) = [];
        end

    end

    thresh = A_edges;

    % Final thresholding
    A_thresh = A_orig .* (abs(A_orig) >= thresh); % Vectorized thresholding

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
