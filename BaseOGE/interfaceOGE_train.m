% Interface to ECOClib 0.3
function c=interfaceOGE_train(data1,data2,OGEparams)

data=[data1,data2];
labels=[ones(size(data1,1)),-ones(size(data2,1))]
[c.N,c.b,c.beta,c.gthr, c.idx1, c.idx2]=train_OGE_sparse(data,labels',OGEparams.type,OGEparams)