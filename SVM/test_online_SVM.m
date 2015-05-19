function [y_test] = test_online_SVM (x_test,model)

  y_test=zeros(1,size(x_test,2));
  for r=1:size(x_test,2) 
    Ktst = GrammMatrixMixed(model.x,x_test(:,r),model.sigma); 
    y_test(r)=model.alfa'*Ktst;
  end
    
end
