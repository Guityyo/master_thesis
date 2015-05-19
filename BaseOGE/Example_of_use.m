load Example

lambda=1;

[N,b,beta,gthr,gthr2]=train_OGE_sparse(d',l,lambda,'tikhonov');
[class,arr]=test_OGE_sparse(d',N,b,beta,gthr);

acc=sum(class==l)/length(l);
s=sprintf('The expected value of this test is 0.9950. Test Accuracy: %f',acc);
disp(s)

%data1=d;
%data2=l;
%OGEparams.type=lambda;
%OGEparams='tikhonov';
%c=interfaceOGE_train(data1,data2,OGEparams)