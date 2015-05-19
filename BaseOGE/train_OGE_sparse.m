%% Data is a dxM matrix where each column is a d-dimensional instance and we have M instances in the training set
%   Inputs: labels-> row vector of labels [+1,-1], lambda-> Regularization parameter (ex. 10), type is 'tikhonov' if we want regularized optimization
%               if not defined the untrained LODE is created.
%   Outputs: N-> normal vectors (one per column), b->base points (one per dichotomizer), beta->ensemble weight, gthr->global threshold (mean value)
%
% [N,b,beta,gthr]=train_OGE_sparse(data,labels,lambda,type)


% This is a beta version of the Optimized Geometry-based Ensemble basic
% classifier.
% Copyright 2008, 2009 Oriol Pujol and David Masip. 
% This software is distributed under the terms of the GNU General Public License


function [N,b,beta,gthr, idx1, idx2]=train_OGE_sparse(data,labels,lambda,type)

OVERRIDE=0;

s=sprintf('Computing CBP...\n');
disp(s);drawnow;
[idx1, idx2,log] = edgesGN(data,labels'); 
N=data(:,idx2)-data(:,idx1);
b=0.5*(data(:,idx1)+data(:,idx2));
N=N./(repmat(sqrt(sum(N.*N))+eps , size(N,1),1));

if OVERRIDE==0
    if size(N,2)>8000
        s=sprintf('>>>> WARNING: High complexity problem. The computation of the model can take a long time. Switching to a linear model or preemtive prunning is advised\n');
        disp(s);
        s=sprintf('>>>> To force the code to run change the varible OVERRIDE to 1 in "train_OGE_sparse.m"\n');
        disp(s);
        return;
    end
end

if strcmp(type,'tikhonov')
    [beta,gthr]=Regularize_OGE_sparse(data,N,b,labels,lambda);
else
    s=sprintf('>>>> WARNING: This implementation only supports tikhonov regularization');
    disp(s);
end
