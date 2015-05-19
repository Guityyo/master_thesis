function [model] = train_online_SVM(x_train,y_train,sigma,lambda,n_iterations)

%% Starting with the formulation
sprintf('\n--------------- SVM with CBP: training ----------------------------------------------\n');


% To pick the same data points
s = RandStream('mt19937ar','Seed',0);
m1 = size(x_train',1); 
ind_SVM=randperm(s,m1);       % pick a single data point at rand

t_ran1=1;

%% initialize values in case no max appears (due to no convergence? or bad parameters?)

for t=1:n_iterations%+it_tot          % iterations over the full data 
    
    if (t_ran1==m1)
        ind_SVM=randperm(s,m1);       % pick a single data point at rand
        t_ran1=1;
    end
    
    index=ind_SVM(t_ran1);
    x_SVM=x_train(:,ind_SVM(t_ran1));    % pick at rand a data point for train    
    t_ran1=1+t_ran1;

    
% Starting of the training 
    if ( t==1 ) % for the initializations
        model.x=x_SVM;
        model.alfa=y_train(index);
        model.sigma=sigma;
        model.lambda=lambda;
        tic;
    else
        
        K_SVM=GrammMatrixMixed(model.x,x_SVM,model.sigma); % computing Kernel of SVM
      
            if (y_train(index)*(model.alfa*K_SVM') < 1)   % to know if it is a support vector 
                
                model.alfa = (1-1/t)*model.alfa+ y_train(index)*K_SVM / (model.lambda*t);
 
                a=find(sum(ismember(model.x(:,:),x_SVM))==length(x_SVM), 1);  
                if (isempty(a)) %to check if element is already in the model   
                    model.x=[model.x x_SVM];
                    model.alfa=[model.alfa;y_train(index)./(model.lambda*t)]; 
                end
         
            else % not a support vector 
                model.alfa = (1-1/t)*model.alfa;
            end
              
    end
    

end
    model.lambda=lambda;

   
end
