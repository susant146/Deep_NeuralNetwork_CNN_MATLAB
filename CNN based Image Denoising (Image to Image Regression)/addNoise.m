function dataOut = addNoise(data)

dataOut = data;
for idx = 1:size(data,1)
   dataOut{idx} = imnoise(data{idx},'salt & pepper');
end

end