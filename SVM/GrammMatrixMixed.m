function K = GrammMatrixMixed(data_1,data_2,sigma) 
N1 = size(data_1,2); 
N2 = size(data_2,2); 
K = zeros([N1 N2]); 
for i = 1:N1 
    for j = 1:N2 
        if i~=j K(i,j) = kernel_test(data_2(:,j),data_1(:,i),sigma); 
        end
    end
end
end
