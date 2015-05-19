function [ model ] = train_dual_kernelized_alternative( labels, data, lambda, K, K2)

[N,n] = size(data); 

cvx_begin 
variables alfa(n,1) u(n) 
minimize (norm(alfa'*K2)+sum(lambda*u'))
subject to 
labels'.*(alfa'*K)>=1-u';
u>=0;
cvx_end

model.alfa=alfa;
model.error=u;
model.terme1=norm(alfa'*K2);
model.terme2=sum(lambda*u');
end

