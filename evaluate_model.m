function [accuracy] = evaluate_model (y,y_pred)

acc=0;
for i=1:length(y)
   if(y(i)==sign(y_pred(i)))
      acc=acc+1; 
   end
end
accuracy=100*acc/length(y);

end
