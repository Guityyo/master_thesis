% This is a beta version of the Optimized Geometry-based Ensemble basic
% classifier.
% Copyright 2008, 2009 Oriol Pujol and David Masip. 
% This software is distributed under the terms of the GNU General Public License
%% Test function for Global LODE.
% The ouput takes values [-1,1]. acc_marg is the ensemble value \Pi(x)
% data is dxM matrix of test instances (one per column)
function [clase,acc_marg]=test_OGE_sparse(data,N,b,beta,gthr)

acc_marg=0;
for j=1:size(b,2) %% For each hyperplane
    tmp_marge=(data-repmat(b(:,j),1,size(data,2)))'*N(:,j);
    acc_marg=acc_marg+beta(j)*((tmp_marge>0)*2-1);
end
clase=((acc_marg>gthr)*2-1);
