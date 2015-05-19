function [model] = train_onlineCBP_SVM(x_train,y_train,x_OGE,lambda,total_iterations)

%% Starting with the formulation
s=sprintf('\n--------------- SVM with CBP: training ----------------------------------------------\n');


% To pick the same data points
s = RandStream('mt19937ar','Seed',0);
m1 = size(x_train',1); 
ind_SVM=randperm(s,m1);       % pick a single data point at rand
% To pick the same CBP points
m2 = size(x_OGE',1); 
ind_CBP=randperm(s,m2);       % pick a single data point at rand

% ON-LINE: at each iteration we pick a rand data point
t_ran1=1;
t_ran2=1;
mu=0;
count=0;
norm_old=0;
norm_old_old=0;


%% initialize values in case no max appears (due to no convergence? or bad parameters?)
found_sigma=0;

check_sigm=15;
it_sub=1000;
it_tot=check_sigm*it_sub;

store_sigma=zeros(check_sigm, 1);
store_norm=zeros(check_sigm, 1);
store_grad=zeros(check_sigm, 1);
count1=1;

for t=1:total_iterations         % iterations over the full data 
    
    if (t_ran2==m2 && t_ran1==m1)
        ind_CBP=randperm(s,m2);       % pick a single data point at rand
        t_ran2=1;
        ind_SVM=randperm(s,m1);       % pick a single data point at rand
        t_ran1=1;
    elseif (t_ran1==m1)
        ind_SVM=randperm(s,m1);       % pick a single data point at rand
        t_ran1=1;
    elseif (t_ran2==m2)
        ind_CBP=randperm(s,m2);       % pick a single data point at rand
        t_ran2=1;
    end
    
    index=ind_SVM(t_ran1);
    x_SVM=x_train(:,ind_SVM(t_ran1));    % pick at rand a data point for train
    x_CBP=x_OGE(:,ind_CBP(t_ran2));      % pick at rand a CBP point for train
    
    t_ran1=1+t_ran1;
    t_ran2=1+t_ran2;
    
% Starting of the training 
    if ( t==1 ) % for the initializations
        model.x=x_SVM;
        model.alfa=y_train(index);


        d_mu=(log(0.01)/log(2))/15 ;
        model.sigma=1;
        mu=-d_mu;
        tic;
    else
        
   
        K_SVM=GrammMatrixMixed(model.x,x_SVM,model.sigma); % computing Kernel of SVM
        K_CBP=GrammMatrixMixed(model.x,x_CBP,model.sigma); % computing Kernel of CBP

            model.norm=(K_CBP'*K_CBP)*model.alfa;
 
            if (y_train(index)*(model.alfa'*K_SVM) < 1)   % to know if it is a support vector 
                
               
                model.alfa = (model.alfa-model.norm./t)+ y_train(index)*K_SVM / (lambda*t);

                if (~ismember(x_SVM',model.x','rows')) %to check if element is already in the model    
                    model.x=[model.x x_SVM];
                    model.alfa=[model.alfa;y_train(index)./(lambda*t)]; % new alpha y/(lambda*t) (lambda yes or no)
                end
                
            else % not a support vector 
       
                model.alfa = (model.alfa-model.norm/t);
            end
           
            
          % Sigma opt %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          
                  if(t<it_tot)
         
                      if(count==it_sub)
       
                        K_CBP=GrammMatrixMixed(model.x,x_CBP,model.sigma); % computing Kernel of CBP
                        norm_new=model.alfa'*(K_CBP'*K_CBP)*model.alfa;
             
                        %%% Regularization norm %%%
                        norm_new=norm_new+0.9*norm_old;
                        grad_new=norm_old-norm_new;
                        %%%
                      
                        % Selecting best sigma %%%%%%%%%%%%%%%%%%%%%%%%%%
                        if(norm_new<norm_old && norm_old>norm_old_old )%&& grad_old<=0 && grad_new>= 0 )
                           found_sigma=model.sigma;
                           found_alfa=model.alfa;
                           found_x=model.x;
                        end

                        
                        store_sigma(count1)=model.sigma;
                        store_norm(count1)=norm_new;
                        store_grad(count1)=grad_new;
                        
                        norm_old_old=norm_old;
                        norm_old=norm_new;
                        
                        count=0;
                        count1=count1+1; 
                        
                        % updating of sigma %
                        mu=mu-d_mu;
                        model.sigma=2^(-mu);

                      end 
                        
            count=count+1; 
                  end 
                  
          if (t==it_tot)
            toc  
            model.sigma=found_sigma; 
            model.alfa=found_alfa;
            model.x=found_x;
          end
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    

end
    model.lambda=lambda;

   
end
