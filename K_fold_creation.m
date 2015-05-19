
function [finalTRAIN,finalTEST]= K_fold_creation(data,K)

[rows,colms]=size(data);

NUM=round(colms/K);           % We divide by the number of folders K
IND=linspace(1,colms,colms);  % We store all the index from data

%% First we will store the indices for each folder
index=[];
for i=1:K
   
    
    if i==K     % For the last folder we just store the last indices
   
    indexLAST=IND;
    
    else  
        
    [rowsIND,colmsIND]=size(IND);
    RandNUM=randperm(colmsIND);
    RandNUM=RandNUM(1:NUM);
    RandNUM=sort(RandNUM,'descend');
    lenR=length(RandNUM);
    INDnums=[];
    len=0;    
        
    for j=1:NUM
       nums=IND(RandNUM(j));
       INDnums=[INDnums;nums];
       IND(RandNUM(j))=[];  % We erase the indices stored
    end
    index=[index INDnums];
    end
end    
    
%% Now we store the indices for training and testing for each folder

finalTRAIN={};
finalTEST={};
for i=1:K
TRAIN=[];
TEST=[];
for j=1:K
    if j==K
        if i==j
         TEST=[TEST indexLAST];     
        else    
         TRAIN=[TRAIN indexLAST]; 
        end
    else    
        if i==j
         TEST=[TEST index(:,j)'];    
        else    
         TRAIN=[TRAIN index(:,j)'];
        end  
    end
end  

finalTRAIN{i}=TRAIN;  % We store train indices in cell
finalTEST{i}=TEST;    % We store test indices in cell

end
end





