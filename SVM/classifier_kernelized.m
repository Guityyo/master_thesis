function [ y ] = classifier_kernelized(data,labels,nu,sigma, x )
y = 0; 
for j = 1:size(data,2) 
  y=y+ (nu(j).*labels(j))*kernel_test(data(:,j),x,sigma);
end

end

