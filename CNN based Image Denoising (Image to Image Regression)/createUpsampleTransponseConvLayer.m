function out = createUpsampleTransponseConvLayer(factor,numFilters)

filterSize = 2*factor - mod(factor,2); 
cropping = (factor-mod(factor,2))/2;
numChannels = 8;

out = transposedConv2dLayer(filterSize,numFilters, ... 
    'NumChannels',numChannels,'Stride',factor,'Cropping',cropping);
end