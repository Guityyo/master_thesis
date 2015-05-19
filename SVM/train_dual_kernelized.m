function [ nu ] = train_dual_kernelized( labels, data, lambda, K)
n = size(data,2); 
m = size(data,1); 
cvx_begin 
variables nu(n,1) 
Q=diag(labels)*K*diag(labels);
size(Q)
maximize( nu'*ones(n,1) - (1/2)*nu'*Q*nu ) 
subject to 
nu'*labels == 0 ;
zeros(n,1) <= nu ;
nu <= lambda.*ones(n,1) ;
cvx_end
end

