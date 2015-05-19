function K = GrammMatrix(x1,x2,sigma) 
N1 = size(x1,2); 
N2 = size(x2,2); 
K = ones([N1 N2]);
for i = 1:N1 
    for j = 1:N2 
        if i~=j 
            K(i,j) = kernel_test(x2(:,j),x1(:,i),sigma); 
        end
    end
end
end
