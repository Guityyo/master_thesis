% This is a beta version of the Optimized Geometry-based Ensemble basic
% classifier.
% Copyright 2008, 2009 Oriol Pujol and David Masip. 
% This software is distributed under the terms of the GNU General Public License



function [NodeA, NodeB] = edgesGN_Sokal(X,labels)

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
in=0;
for i = 1:N1
    for j= 1:N2
        %Check the GN condition
        for k = 1:N
            if and((k ~= i),(k~=j))
                A2 = D2(k,label1(i));
                B2 = D2(k,label2(j));
                C2 = D2(label2(j),label1(i));
                if A2+B2<C2
                   break;
                end
            end           
        end
        if k == N
            NodeA = [NodeA i];
            NodeB = [NodeB j];            
        end
        in=0;
    end
end
NodeA=label1(NodeA);
NodeB=label2(NodeB);

