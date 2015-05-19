% This is a beta version of the Optimized Geometry-based Ensemble basic
% classifier.
% Copyright 2008, 2009 Oriol Pujol and David Masip. 
% This software is distributed under the terms of the GNU General Public License



function [NodeA, NodeB, log] = edgesGN(X,labels)

[D N] = size(X);

labelval=unique(labels);
label1 = find(labels == labelval(1));
label2 = find(labels == labelval(2));

N1 = size(label1,2);
N2 = size(label2,2);

%precomputing the distances
D = L2_distance(X,X);
D2 = D.*D;

NodeA = [];
NodeB = [];

for i = 1:N1
    for j= 1:N2
        %Check the GN condition
        for k = 1:N

            if and((k ~= i),(k~=j))
                A = D(k,label1(i));
                B = D(k,label2(j));
                C = D(label2(j),label1(i));

                A2 = D2(k,label1(i));
                B2 = D2(k,label2(j));
                C2 = D2(label2(j),label1(i));

                tmp = ((B2-A2-C2)/(-2*A*C));
                dist = A2*(1-tmp*tmp)+(A*tmp-C/2).^2;

                if dist < (C/2).^2
                   break; 
                end
            end           
        end
        log(i,j) = k;
        if k == N
            NodeA = [NodeA i];
            NodeB = [NodeB j];            
        end

    end
end
NodeA=label1(NodeA);
NodeB=label2(NodeB);

