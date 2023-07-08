% This is an example CNN artchitecture designed for image denoising.
% The program uses 'denoisedImageDatastore' for automatically creating
% training image patches of 'input' and 'response' data. 
% input = Reference/original image patch.
% response = Noisy image patch.
% -------------------------------------------------------------------
clear 
close all
clc

%% Step - 01: Load image data. 
% We will be using matlab inbuilt image data for denosing.
setDir = fullfile(toolboxdir('images'),'imdata');
imds = imageDatastore(setDir,'FileExtensions',{'.jpg'});

% Preview a sample of data.
img = preview(imds);
figure, imshow(img, [])

%% Step - 02: Generate training data using denoisingImageDatastore
dnds = denoisingImageDatastore(imds,...
'PatchesPerImage',512,...
'PatchSize',50,...
'GaussianNoiseLevel',[0.01 0.1],...
'ChannelFormat','grayscale');
% From each image 512-patches will be extracted: Each patch is of size
% 50*50. imds_totImages = readall(imds)
% imds_totImages = 38. 
% So total input and response pair in 'dnds' will be 
% 'imds_totImages*'PatchesPerImage',512'
% Can be checked using dnds_totImData = readall(dnds);

%% Step - 03: Design Network Architecture. 
% Note: As patch size in training data is 50*50 the input Layer will be
% 50*50. Moreover we are creating a image to image regression network thus
% output will also be 50*50. 

% Layers = [imageInputLayer([50,50,1])
%     convolution2dLayer(3, 64, 'Stride',1,'Padding',[1,1,1,1])
%     reluLayer
%     
%     convolution2dLayer(3, 64, 'Stride',1,'Padding',[2,2,2,2])
%     reluLayer
%     convolution2dLayer(3, 64, 'Stride',1,'Padding',[1,1,1,1])
%     reluLayer
%     
%     convolution2dLayer(3,1)
%     regressionLayer]; 

% Note: The available DnCNN architecture can also be used here. 
% In case of using DnCNN for training use the following code:
% layers = dnCNNLayers('NetworkDepth', 20); %default network depth = 20.
% Total network layes will be = 59.
layers = dnCNNLayers('NetworkDepth', 20);

% Network Training Options:
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 5, ...
    'ValidationFrequency', 5, ...
    'InitialLearnRate', 1e-4, 'plots','training-progress');

%% Step - 04: Train the Network.
[net, traininfo] = trainNetwork(dnds, layers, options);

%% Step-05: Testing CNN Network for Image Denoising.
imR = imread('cameraman.tif');
imN = imnoise(imR, "gaussian", 0.1, 0.001);

% Denoise Image using CNN.
imD = denoiseImage(imN,net);

