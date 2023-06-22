% -- First Neural Network (MNIST Hand-Writing data) for Classification --
% This is a Neural Network to classify handwritten images of digits 
% from the MNIST dataset. This is another prototypical hello world code 
% that utilizes the Keras-styled neural network writing capabilities 
% of MATLAB.
clear 
close all
clc

% ------- LOAD Dataset for Image Classficication ---------
% Inbuit Folder containg MNIST dataset of hand-writing digit images are
% available in MATLAB folder path ----- Read the FOLDER -----
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset'); % Full path is mentioned here.
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Displaying a Single Randomly Choosen Image-----
% Note: Total Images = 10000, Each folder (0 to 10) = 1000 Images.
figure, 
imshow(imds.Files{100});
test_im = imread(imds.Files{100});
[row, col] = size(test_im);

% Count the total number of Classes: 
labelCount = countEachLabel(imds); % Each folder consist of 1000 images

% ---- Split Images into TRAINING & TESTING(VALIDATION) Sets -----
% Total Trainig Images = 750
% Total Test Images = 250  for each class.
numTrainFiles = 750;
[imdsTrain,imdsValidation]= splitEachLabel(imds,numTrainFiles,'randomize');

% -------- Define Neural Network Architecture --------------
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% ---- Train the Deep Neural Network Architecture ----------
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% --- Using Above training parameters Train the DNN ----
net = trainNetwork(imdsTrain,layers,options);

% ---- Compute the NETWEORk Accuracy -----------
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);

% ----- Classify and Display Image --------------
% Read an image for classification.
I = imread("image3276.png");
inputSize = net.Layers(1).InputSize;
I = imresize(I,inputSize(1:2));

label = classify(net,I);
figure
imshow(I)
title(string(label))
