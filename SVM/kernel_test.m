function [ k ] = kernel_test( x_i, x_j, sigma )
d=L2_distance(x_i, x_j);
k = exp((-(d.^2))/(2*sigma^2));
end

