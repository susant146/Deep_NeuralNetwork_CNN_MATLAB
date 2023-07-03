% Image-to-Image Regression for Denoising.
% AutoEncoder and AutoDecoder for image densoising.
% This example is used for 'Salt&Pepper' Noise removal.

clear
close all
clc


digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Specify a large read size
imds.ReadSize = 500;

% Specify the same noise to be added using rng.
rng(0)

% shuffle the digit data prior to training.
imds = shuffle(imds);

% divide imds into three image datastores containing pristine images 
% for training, validation, and testing.
[imdsTrain,imdsVal,imdsTest] = splitEachLabel(imds,0.95,0.025);
% splitEachLabel(imdsout,0.7,0.15,0.15,'randomized');

% Use 'transform' function to addNoise in various dataset.
dsTrainNoisy = transform(imdsTrain,@addNoise);
dsValNoisy = transform(imdsVal,@addNoise);
dsTestNoisy = transform(imdsTest,@addNoise);

% Combine the respective noisy and reference/original image.
dsTrain = combine(dsTrainNoisy,imdsTrain);
dsVal = combine(dsValNoisy,imdsVal);
dsTest = combine(dsTestNoisy,imdsTest);

% Image Preprocessing: Resizing = [32,32] and single.
dsTrain = transform(dsTrain,@commonPreprocessing);
dsVal = transform(dsVal,@commonPreprocessing);
dsTest = transform(dsTest,@commonPreprocessing);

% Increase TrainData using DataAugmentation.
dsTrain = transform(dsTrain,@augmentImages);

% ----- exampleData = preview(dsTrain); ----
% This is used for data-preview and also can be used for debaugging.
exampleData = preview(dsTrain);
inputs = exampleData(:,1);
responses = exampleData(:,2);
minibatch = cat(2,inputs,responses);
montage(minibatch','Size',[8 2])
title('Inputs (Left) and Responses (Right)')

% Read all training/Testing/validation data.
% X = readall(dsTrain);

% Define the CNN (AutoEncoder and AutoDecoder) Architecture of Denoising.
imageLayer = imageInputLayer([32,32,1]);
% AutoEncoder.
encodingLayers = [ ...
    convolution2dLayer(3,16,'Padding','same'), ...
    reluLayer, ...
    maxPooling2dLayer(2,'Padding','same','Stride',2), ...
    convolution2dLayer(3,8,'Padding','same'), ...
    reluLayer, ...
    maxPooling2dLayer(2,'Padding','same','Stride',2), ...
    convolution2dLayer(3,8,'Padding','same'), ...
    reluLayer, ...
    maxPooling2dLayer(2,'Padding','same','Stride',2)];

% AutoDecoder.
decodingLayers = [ ...
    createUpsampleTransponseConvLayer(2,8), ...
    reluLayer, ...
    createUpsampleTransponseConvLayer(2,8), ...
    reluLayer, ...
    createUpsampleTransponseConvLayer(2,16), ...
    reluLayer, ...
    convolution2dLayer(3,1,'Padding','same'), ...
    clippedReluLayer(1.0), ...
    regressionLayer];    

%Concatenate the image input layer, the encoding layers, and the decoding 
% layers to form the convolutional autoencoder network architecture.
layers = [imageLayer,encodingLayers,decodingLayers];

% Analyze the network for error debugging:
analyzeNetwork(layers)

% Define Training Options.
options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'MiniBatchSize',imds.ReadSize, ...
    'ValidationData',dsVal, ...
    'Shuffle','never', ...
    'Plots','training-progress', ...
    'Verbose',false, 'ExecutionEnvironment','cpu');

% Train Network.
net = trainNetwork(dsTrain,layers,options);

% Evaluate the Performance of the Denoising Network
ypred = predict(net,dsTest);

% Preview the noisy and predicted image.
inputImageExamples = preview(dsTest);
montage({inputImageExamples{1},ypred(:,:,:,1)});

ref = inputImageExamples{1,2};
originalNoisyImage = inputImageExamples{1,1};
psnrNoisy = psnr(originalNoisyImage,ref)
psnrDenoised = psnr(ypred(:,:,:,1),ref)

%% --- Sample Code -----------
% inputData=fileDatastore(fullfile('inputData'),'ReadFcn',@load,'FileExtensions','.mat');
% targetData=fileDatastore(fullfile('targetData'),'ReadFcn',@load,'FileExtensions','.mat');
% 
% inputDatat = transform(inputData,@(data) rearrange_datastore(data));
% targetDatat = transform(targetData,@(data) rearrange_datastore(data));
% 
% trainData=combine(inputDatat,targetDatat);
% 
% % here I defined my network architecture
% 
% % here I defined my training options
% 
% net=trainNetwork(trainData, Layers, options);
% 
% 
% function image = rearrange_datastore(data)
% image=data.C;
% image= {image};
% end