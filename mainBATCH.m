 
clear all
close all
clc
warning off

addpath('SVM');
addpath('SVM-CBP');
addpath('BaseOGE');
addpath('cvx');
addpath('datasets');
%cvx_setup

% Parameters to tune at the beginning %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nf=5; % number of folders for cross-validation
lamb = [0.01,0.1,1,10,100,1000]; % lambda values to check
sigm = [0.05,0.2,0.5,0.75,1]; % sigma values to check (only for SVM)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataset={'AustralianCreditApproval','BankAu','Fertility','Ionosphere','LiverDisorders','Pima','QSARbio','StatlogHeart','SPECTFHeart','WhoSaCus'}; % real datasets 

for dt = 1 : length(dataset) % For every dataset
    
% Loading the data
name=dataset{dt};
data = xlsread(name);
fprintf('\n\t-----------------------------------------------------------\n');
fprintf('\n\t\t DATASET    %s\n', name );
fprintf('\n\t-----------------------------------------------------------\n');

% Divide in data and labels� 
labels=data(:,size(data,2));
data=data(:,1:size(data,2)-1);
labels(labels==0)=-1;

% Normalize the data
[data] = normalizeData(data');

% Divide into train and test 
[fil,col]=size(data);
del=round(col*80/100);
ind_ran=randperm(col,col);       

x_test=data(:,ind_ran(del+1:col));
y_test=labels(ind_ran(del+1:col));
data=data(:,ind_ran(1:del));
labels=labels(ind_ran(1:del));

% Create the folders for posterior cross-validation
[train_ind,val_ind]= K_fold_creation(data,Nf);

save(strcat('datasets_split/data_',name));
load(strcat('datasets_split/data_',name));

%% SVM-CBP TRAIN AND TEST

z=clock; % To see the time spent with the training and test

total_time=Nf*length(lamb)+Nf*length(lamb)*length(sigm);
time=0;

Accuracy_total_train_1=zeros(Nf, length(lamb));
Accuracy_total_mean_train=zeros(Nf,1);
Accuracy_total_val_1=zeros(Nf, length(lamb));
Accuracy_total_mean_val=zeros(Nf,1);

best_sigma_1=0;
best_lambda_1=0;
best_mean_train_1=0;
best_folds_train_1=[];
best_mean_val_1=0;
best_val_1=[];
best_folds_val_1=[];

for l=1:length(lamb) % Starting the cross-validation
 
 lambda=lamb(l);
 Accuracy_folds_train=zeros(Nf, 1); % To store the accuracies
 Accuracy_folds_val=zeros(Nf, 1);

for n=1:Nf % For every lambda the Nf folders

% Selecting the folder
x_train=data(:,cell2mat(train_ind(n)));
y_train=labels(cell2mat(train_ind(n)));
x_validation=data(:,cell2mat(val_ind(n)));
y_validation=labels(cell2mat(val_ind(n)));

% Getting the OGE points of the training set   
lambOGE=0.01;  % of the OGE
[N,x_CBP,beta,gthr,gthr2]=train_OGE_sparse(x_train,y_train,lambOGE,'tikhonov');

% Tunning of the sigma
terme1_past=0;
terme1_present=0;
terme1_past_past=0;
model_present=0;
best_sigma=0.1;

% Sigma decreasing step
d_mu=(log(0.01)/log(2))/10;
mu=-d_mu;
sigma=2^(-mu);

% Automatic sigma optimization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while ( terme1_past<terme1_present )
K = GrammMatrix(x_train,sigma);   
K2 = GrammMatrixMixed(x_train,x_CBP,sigma);
[model] = train_dual_kernelized_alternative(y_train, x_train, lambda, K, K2);
best_sigma=sigma;

% Regularization norm %%%%%%s
terme1_present=model.terme1+0.9*terme1_past;

% Sigma decreasing step %%%%
mu=mu-d_mu;
sigma=2^(-mu);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

terme1_past=terme1_present;
terme1_past_past=terme1_past;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Training the model
K = GrammMatrix(x_train,x_train,best_sigma);   
K2 = GrammMatrixMixed(x_train,x_CBP,best_sigma);
[model] = train_dual_kernelized_alternative(y_train, x_train, lambda, K, K2);

 
% Training accuracy
acc=0;
class_pred=zeros(length(y_train), 1);
for m = 1:size(x_train,2) 
    class_pred(m) = classifier_kernelized_alternative(x_train,y_train,model.alfa,best_sigma,x_train(:,m));
    if(y_train(m)==sign(class_pred(m)))
        acc=acc+1;
    end
end 
Accuracy_folds_train(n)=100*acc/length(y_train);


% Validation
acc=0;
class_pred=zeros(length(y_validation), 1);
for m = 1:size(x_validation,2) 
    class_pred(m) = classifier_kernelized_alternative(x_train,y_train,model.alfa,best_sigma,x_validation(:,m));
    if(y_validation(m)==sign(class_pred(m)))
        acc=acc+1;
    end
end 
Accuracy_folds_val(n)=100*acc/length(y_validation);



% Time processing
time=time+1;

fprintf('\n------------------------------------------------------------------\n');
progress=num2str(100*time/total_time);
progres=strcat('Progress SVM-CBP: ',progress,'%%\n');
fprintf(progres);
fprintf('\n------------------------------------------------------------------\n');


end
 
Accuracy_total_train_1(:,l)=Accuracy_folds_train;
Accuracy_total_mean_train(l)=mean(Accuracy_folds_train);
Accuracy_total_val_1(:,l)=Accuracy_folds_val;
Accuracy_total_mean_val(l)=mean(Accuracy_folds_val);

% Selecting the best fold
if(best_mean_val_1<mean(Accuracy_folds_val))
best_sigma_1=best_sigma;
best_lambda_1=lambda;
best_mean_train_1=mean(Accuracy_folds_train);
best_folds_train_1=Accuracy_folds_train;
best_mean_val_1=mean(Accuracy_folds_val);
best_folds_val_1=Accuracy_folds_val;
end

end

%% Testing the model

% Getting the OGE points of the training set    
lambOGE=0.01;  % of the OGE
[N,x_CBP,beta,gthr,gthr2]=train_OGE_sparse(data,labels,lambOGE,'tikhonov');

% Starting with the setting of the sigma authomatically and training
K = GrammMatrix(data,data,best_sigma_1);   
K2 = GrammMatrixMixed(data,x_CBP,best_sigma_1);
[model] = train_dual_kernelized_alternative(labels, data, best_lambda_1, K, K2);


% Testing the model
class_pred=zeros(length(y_test), 1);
acc=0;
for m = 1:length(y_test)
    class_pred(m) = classifier_kernelized_alternative(data,labels,model.alfa,best_sigma_1,x_test(:,m));
    if(y_test(m)==sign(class_pred(m)))
        acc=acc+1;
    end
end 
Accuracy_test_1=100*acc/length(y_test);


% Total time
timeM1=etime(clock,z);
 



%% SVM TRAINING AND TESTING

 z=clock;% To see the time spent with the training and test

% Starting 
 best_sigma_2=0;
 best_lambda_2=0;
 best_mean_train_2=0;
 best_folds_train_2=[];
 best_mean_val_2=0;
 best_val_2=[];
 best_folds_val_2=[];
 
Accuracy_total_train_2=zeros(Nf, length(lamb)+length(sigm));
Accuracy_total_val_2=zeros(Nf, length(lamb)+length(sigm));
num=1;

for l=1:length(lamb) % For every lambda
lambda=lamb(l);

for s=1:length(sigm)  % For every sigma
sigma=sigm(s);

Accuracy_folds_train=zeros(Nf, 1); % To store the accuracies
Accuracy_folds_validation=zeros(Nf, 1);

for n=1:Nf % For every folder

% Selecting the folder
x_train=data(:,cell2mat(train_ind(n)));
y_train=labels(cell2mat(train_ind(n)));
x_validation=data(:,cell2mat(val_ind(n)));
y_validation=labels(cell2mat(val_ind(n)));

% Calculing the Gram matrix
K = GrammMatrix(x_train,x_train,sigma);

% Training the SVM+kernel 
nu  = train_dual_kernelized(y_train, x_train, lambda, K);

% Train accuracy
acc=0;
class_label=zeros(length(y_train),1);
for cl = 1:length(y_train)
class_label(cl) = classifier_kernelized(x_train,y_train,nu,sigma,x_train(:,cl));
    if(y_train(cl)==sign(class_label(cl)))
        acc=acc+1;
    end
end 
Accuracy_folds_train(n)=100*acc/length(y_train);


% Validation
acc=0;
class_label=zeros(length(y_validation),1);
for cl = 1:length(y_validation)
 class_label(cl) = classifier_kernelized(x_train,y_train,nu,sigma,x_validation(:,cl));
    if(y_validation(cl)==sign(class_label(cl)))
        acc=acc+1;
    end
end 
Accuracy_folds_validation(n)=100*acc/length(y_validation);


% Progress of the computation
fprintf('\n------------------------------------------------------------------\n');
progress=num2str(100*time/total_time);
progres=strcat('Progress SVM: ',progress,'%%\n');
fprintf(progres);
fprintf('\n------------------------------------------------------------------\n');

time=time+1;
end

Accuracy_total_train_2(:,num)=Accuracy_folds_train;
Accuracy_total_val_2(:,num)=Accuracy_folds_val;
num=num+1;

% Selecting the best folders
if(best_mean_val_2<mean(Accuracy_folds_val))
best_sigma_2=sigma;
best_lambda_2=lambda;
best_mean_train_2=mean(Accuracy_folds_train);
best_folds_train_2=Accuracy_folds_train;
best_mean_val_2=mean(Accuracy_folds_val);
best_folds_val_2=Accuracy_folds_val;
end

end

end

%% Testing the SVM
% Calculing the Gram matrix
K = GrammMatrix(data,data,best_sigma_2); 

% Training the SVM+kernel 
nu  = train_dual_kernelized(labels, data, best_lambda_2, K);

% Training testing the SVM+kernel
acc=0;
class_label=zeros(length(y_test),1);
for cl = 1:length(y_test)
class_label(cl) = classifier_kernelized(data,labels,nu,best_sigma_2,x_test(:,cl));
    if(y_test(cl)==sign(class_label(cl)))
        acc=acc+1;
    end
end 
Accuracy_test_2=100*acc/length(y_test);


% Total time
timeM2=etime(clock,z);


% Print results in a document
fileID = fopen(strcat('results_Batch/',name,'_results.txt'),'w');
fprintf(fileID,'\n');
fprintf(fileID,'----------- SVM + OGE ----------------------------------\n');
fprintf(fileID,'total time %12.8f\n',timeM1);
fprintf(fileID,'lambda %12.8f\n',best_lambda_1);
fprintf(fileID,'sigma %12.8f\n',best_sigma_1);
fprintf(fileID,'\n');
fprintf(fileID,'-- SVM + OGE ( train  ) --\n');
fprintf(fileID,'mean folds %12.8f\n',best_mean_train_1);
fprintf(fileID,'%6.2f \n',best_folds_train_1);
fprintf(fileID,'\n');
fprintf(fileID,'-- SVM + OGE ( validation ) --\n');
fprintf(fileID,'mean folds %12.8f\n',best_mean_val_1);
fprintf(fileID,'%6.2f \n',best_folds_val_1);
fprintf(fileID,'\n');
fprintf(fileID,'-- SVM + OGE ( test ) --\n');
fprintf(fileID,'accuracy test %12.8f\n',Accuracy_test_1);
fprintf(fileID,'\n');
fprintf(fileID,'-------------- SVM -------------------------------------\n');
fprintf(fileID,'total time %12.8f\n',timeM2);
fprintf(fileID,'sigma %12.8f\n',best_sigma_2);
fprintf(fileID,'lambda %12.8f\n',best_lambda_2);
fprintf(fileID,'\n');
fprintf(fileID,'-- SVM ( train ) --\n');
fprintf(fileID,'mean folds %12.8f\n',best_mean_train_2);
fprintf(fileID,'%6.2f \n',best_folds_train_2);
fprintf(fileID,'\n');
fprintf(fileID,'-- SVM ( validation ) --\n');
fprintf(fileID,'mean folds %12.8f\n',best_mean_val_2);
fprintf(fileID,'%6.2f \n',best_folds_val_2);
fprintf(fileID,'\n');
fprintf(fileID,'-- SVM ( test ) --\n');
fprintf(fileID,'accuracy test%12.8f\n',Accuracy_test_2);
fclose(fileID);




end