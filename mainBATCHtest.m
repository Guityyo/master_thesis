 
clear all
close all
clc
warning off

addpath('SVM');
addpath('BaseOGE');
addpath('cvx');
addpath('toy_datasets');
addpath('datasets');
%cvx_setup

% Parameters to tune at the beginning %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nf=5; % number of folders for cross-validation
lambda = [0.01,0.1,1,10,100,1000]; % lambda values to check
sigma = [0.05,0.2,0.5,0.75,1]; % sigma values to check (only for SVM)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%dataset={'AustralianCreditApproval','BloodTransf','BankAu','Fertility','LiverDisorders','Pima','QSARbio','StatlogHeart','SPECTFHeart2','WhoSaCus'}; % real datasets 
dataset={'Fertility'}; % real datasets  'SteelPF'

for dt = 1 : length(dataset)%10

    
% Loading the data

name=dataset{dt}
data = xlsread(name);

% Divide in data and labels
 
labels=data(:,size(data,2));
data=data(:,1:size(data,2)-1);
labels(labels==0)=-1;

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

 
 
%% FIRST OUR MODEL TRAINING AND TESTING

z=clock; % To see the time spent with the training and test

total_time=Nf*lamblen+Nf*lamblen*sigmlen;
time=0;

Accuracy_total_train_1=zeros(Nf, lamblen);
Accuracy_total_mean_train=zeros(Nf,1);
Accuracy_total_val_1=zeros(Nf, lamblen);
Accuracy_total_mean_val=zeros(Nf,1);

best_sigma_1=0;
best_lambda_1=0;
best_mean_train_1=0;
best_folds_train_1=[];
best_mean_val_1=0;
best_val_1=[];
best_folds_val_1=[];

for i=1:lamblen % Starting the cross-validation
 
 l=lambda(i);
 Accuracy_folds_train=zeros(Nf, 1); % To store the accuracies
 Accuracy_folds_val=zeros(Nf, 1);

for j=1:Nf % For every lambda the Nf folders

% Selecting the folder
x_train=data(:,cell2mat(train_ind(j)));
y_train=labels(cell2mat(train_ind(j)));
x_validation=data(:,cell2mat(val_ind(j)));
y_validation=labels(cell2mat(val_ind(j)));

x_train_OGE=x_train;
y_train_OGE=y_train;


% Getting the OGE points of the training set   
lambOGE=0.01;  % of the OGE
[N,x_OGE,beta,gthr,gthr2]=train_OGE_sparse(x_train_OGE,y_train_OGE,lambOGE,'tikhonov');

% Tunning of the sigma
terme1_past=0;
terme1_present=0;
terme1_past_past=0;
sigmaOGE=0.95;
model_present=0;
best_sigma=0.1;

% Sigma decreasing step
d_mu=(log(0.01)/log(2))/10;%15 ;
mu=-d_mu;
model.sigma=2^(-mu);
t=1;
%while ( t<=10 ) %Need to change it 
while ( terme1_past<terme1_present )
K = GrammMatrix(x_train,x_train,sigmaOGE);   
K2 = GrammMatrix(x_train,x_OGE,sigmaOGE);
[model] = train_dual_kernelized_alternative(y_train, x_train, l, K, K2);
% Regularization %%%%
terme1_present=model.terme1+0.9*terme1_past
%model.sigmaOGE=sigmaOGE;
%if(terme1_past>terme1_present && terme1_past_past<terme1_past)
best_sigma=sigmaOGE
%end
% Sigma decreasing step %%%%
%sigmaOGE=sigmaOGE-0.05 % decreasing sigma step
mu=mu-d_mu;
sigmaOGE=2^(-mu)
%%%%%%%%%%%%%%%%%%%%%
terme1_past=terme1_present;
terme1_past_past=terme1_past;
t=t+1
end


% Getting the OGE points of the training set    
%lambOGE=0.01;  % of the OGE
%[N,x_OGE,beta,gthr,gthr2]=train_OGE_sparse(x_train,y_train,lambOGE,'tikhonov');

% Starting with the setting of the sigma authomatically and training
K = GrammMatrix(x_train,x_train,best_sigma);   
K2 = GrammMatrix(x_train,x_OGE,best_sigma);
[model] = train_dual_kernelized_alternative(y_train, x_train, l, K, K2);


 
% Training the model
acc=0;
class_pred=zeros(length(y_train), 1);
for m = 1:size(x_train,2) 
    class_pred(m) = classifier_kernelized_alternative(x_train,y_train,model.alfa,best_sigma,x_train(:,m));
    if(y_train(m)==sign(class_pred(m)))
        acc=acc+1;
    end
end 
Accuracy_folds_train(j)=100*acc/length(y_train);

% Validating the model
acc=0;
class_pred=zeros(length(y_validation), 1);
for m = 1:size(x_validation,2) 
    class_pred(m) = classifier_kernelized_alternative(x_train,y_train,model.alfa,best_sigma,x_validation(:,m));
    if(y_validation(m)==sign(class_pred(m)))
        acc=acc+1;
    end
end 
Accuracy_folds_val(j)=100*acc/length(y_validation);



% Time processing
time=time+1;

fprintf('\n------------------------------------------------------------------\n');
progress=num2str(100*time/total_time);
progres=strcat('Progress SVM+CBP:\t',progress,'%%\n');
fprintf(progres);
fprintf('\n------------------------------------------------------------------\n');


end
 
Accuracy_total_train_1(:,i)=Accuracy_folds_train

Accuracy_total_mean_train(i)=mean(Accuracy_folds_train)

Accuracy_total_val_1(:,i)=Accuracy_folds_val
 
Accuracy_total_mean_val(i)=mean(Accuracy_folds_val)

if(best_mean_val_1<mean(Accuracy_folds_val))
best_sigma_1=best_sigma;
best_lambda_1=l;
best_mean_train_1=mean(Accuracy_folds_train);
best_folds_train_1=Accuracy_folds_train;
best_mean_val_1=mean(Accuracy_folds_val);
best_folds_val_1=Accuracy_folds_val;
end

end

%% Testing the model

% Getting the OGE points of the training set    
lambOGE=0.01;  % of the OGE
[N,x_OGE,beta,gthr,gthr2]=train_OGE_sparse(data,labels,lambOGE,'tikhonov');

% Starting with the setting of the sigma authomatically and training
K = GrammMatrix(data,data,best_sigma_1);   
K2 = GrammMatrix(data,x_OGE,best_sigma_1);
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
Accuracy_test_1=100*acc/length(y_test)


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
 
Accuracy_total_train_2=zeros(Nf, lamblen+sigmlen);
Accuracy_total_val_2=zeros(Nf, lamblen+sigmlen);
num=1;

for i=1:lamblen % Starting the cross-validation
l=lambda(i);

for q=1:sigmlen     
s=sigma(q);

Accuracy_folds_train=zeros(Nf, 1); % To store the accuracies
Accuracy_folds_validation=zeros(Nf, 1);

for j=1:Nf % For every lambda the Nf folders

% Selecting the folder

x_train=data(:,cell2mat(train_ind(j)));
y_train=labels(cell2mat(train_ind(j)));
x_validation=data(:,cell2mat(val_ind(j)));
y_validation=labels(cell2mat(val_ind(j)));

%x_train=data(cell2mat(train_ind(j)),:);
%y_train=labels(cell2mat(train_ind(j)));
%x_val=data(cell2mat(val_ind(j)),:);
%y_val=labels(cell2mat(val_ind(j)));
%x_train=x_train';
%x_val=x_val';

% Calculing the Gram matrix
K = GrammMatrix(x_train,x_train,s);

% Training the SVM+kernel 
nu  = train_dual_kernelized(y_train, x_train, l, K);


% Training testing the SVM+kernel
acc=0;
class_label=zeros(length(y_train),1);
for cl = 1:length(y_train)
class_label(cl) = classifier_kernelized(x_train,y_train,nu,s,x_train(:,cl));
    if(y_train(cl)==sign(class_label(cl)))
        acc=acc+1;
    end
end 
Accuracy_folds_train(j)=100*acc/length(y_train);


% Testing the SVM+kernel
acc=0;
class_label=zeros(length(y_validation),1);
for cl = 1:length(y_validation)
 class_label(cl) = classifier_kernelized(x_train,y_train,nu,s,x_validation(:,cl));
    if(y_validation(cl)==sign(class_label(cl)))
        acc=acc+1;
    end
end 
Accuracy_folds_validation(j)=100*acc/length(y_validation);


% Progress of the computation
fprintf('\n------------------------------------------------------------------\n');
progress=num2str(100*time/total_time);
progres=strcat('Progress SVM:\t',progress,'%%\n');
fprintf(progres);
fprintf('\n------------------------------------------------------------------\n');

time=time+1;
end

% if dt==1
% % Ploting the dataset and the separtaing hyperplane
% Sigma = num2str(s);
% Lambda = num2str(l);
% a=figure;
% contourf(xi,yi,reshape(Y,size(xi))); hold on
% contour(xi,yi,reshape(Y,size(xi)), [-1 1 ],'linewidth',2,'linecolor',[0.5,0.5,0.5])
% contour(xi,yi,reshape(Y,size(xi)), [0. 0.],'linewidth',3,'linecolor',[0,0,0])
% hold on
% %plot(data(1,find(labels==-1)),data(2,find(labels==-1)),'b*');
% %plot(data(1,find(labels==+1)),data(2,find(labels==+1)),'r*');
% plot(x_OGE(1,:),x_OGE(2,:),'ko','markersize',12,'linewidth',1,'MarkerFaceColor','y');
% plot(x_train(1,y_train==-1),x_train(2,y_train==-1),'ko','markersize',6,'linewidth',1,'MarkerFaceColor','c');
% plot(x_train(1,y_train==+1),x_train(2,y_train==+1),'ko','markersize',6,'linewidth',1,'MarkerFaceColor','m');
% plot(x_val(1,y_val==-1),x_val(2,y_val==-1),'ko','markersize',6,'linewidth',1,'MarkerFaceColor','b');
% plot(x_val(1,y_val==+1),x_val(2,y_val==+1),'ko','markersize',6,'linewidth',1,'MarkerFaceColor','r');
% title(strcat('Sigma = ',Sigma,'        Lambda = ',Lambda)); % To put the title to the image
% filePath = strcat('LAST_RESULTS/',name,'_SVM_',Sigma,'-',Lambda,'.jpg');  % To save the figure
% saveas(a,filePath);%,'jpg') ;
% hold off
% close(a)
% end

Accuracy_total_train_2(:,num)=Accuracy_folds_train
Accuracy_total_val_2(:,num)=Accuracy_folds_val
num=num+1;

if(best_mean_val_2<mean(Accuracy_folds_val))
best_sigma_2=s;
best_lambda_2=l;
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
Accuracy_test_2=100*acc/length(y_test)


% Total time

timeM2=etime(clock,z);


% Print results in a document
fileID = fopen(strcat('LAST_RESULTS/',name,'_results.txt'),'w');
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


fid = fopen(strcat('LAST_RESULTS/',name,'_results_boxplot.txt'),'wt'); % Open for writing
fprintf(fid,'\n----------- batch  SVM-CBP----------------------------------\n');
fprintf(fid,'\taccuracy folds train\n');
for p=1:size(best_folds_train_1,1)
   fprintf(fid, '%d ', best_folds_train_1(p,:));
   fprintf(fid, '\n');
end
fprintf(fid,'\taccuracy folds test\n');
for p=1:size(best_folds_val_1,1)
   fprintf(fid, '%d ', best_folds_val_1(p,:));
   fprintf(fid, '\n');
end
fprintf(fid,'\n----------- batch SVM ----------------------------------\n');
fprintf(fid,'\taccuracy folds train\n');
for p=1:size(best_folds_train_2,1)
   fprintf(fid, '%d ', best_folds_train_2(p,:));
   fprintf(fid, '\n');
end
fprintf(fid,'\taccuracy folds train\n');
for p=1:size(best_folds_val_2,1)
   fprintf(fid, '%d ', best_folds_val_2(p,:));
   fprintf(fid, '\n');
end
fclose(fid);


end