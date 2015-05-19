
function [ dataNorm ] = normalizeData(data)

[rows,colms]=size(data);

for i=1:colms
   MIN=min(data(:,i));
   MAX=max(data(:,i));
   dataNorm(:,i)=(data(:,i)-MIN)/(MAX-MIN);
end


end

