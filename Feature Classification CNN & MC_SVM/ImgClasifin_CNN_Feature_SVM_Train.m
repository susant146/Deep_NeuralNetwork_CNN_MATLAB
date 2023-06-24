% Image Feature extraction and Multiclass Image Classification using SVM
% Extract Image Features Using Pretrained Network
% https://in.mathworks.com/help/deeplearning/ug/extract-image-features-using-pretrained-network.html
% --------- STEPS -------
% Prepare Image data for classification.
% Divide in Training and Testing Data
% Use a pre-trained network for image feature extraction.
% AlexNet, ResNet, GoogleNet or any other Net
% Use multiclass models for support vector machines (SVM)
% Design classifier. 
% Find the accuracy.
% Use the saved 'classifier' for testing various images.
% ---------------------------------------------------------
% Note: The classifier obtained from the pre-trained network would work
% much better compared to other classifiers like:
% Histogram of oriented gradients (HOG)
% Speeded-up robust features (SURF)
% Local binary pattern (LBP).
% -------------------------------------------------------------
% Two '.m' files are created one for testing and another for training.
% -------------------------------------------------------------------
clear 
close all
clc

% Uncomment below block of code for downloading image data.
%% If image not avaiable in local computer then Download image data and
% unzip using the following code.

% Location of the compressed data set
% url = 'https://www.mathworks.com/supportfiles/nnet/data/ExampleFoodImageDataset.zip';
% 
% % Store the output in a temporary folder
% downloadFolder = pwd;
% filename = fullfile(downloadFolder,'ExampleFoodImageDataset.zip');
% 
% % Uncompressed data set
% imageFolder = fullfile(downloadFolder,"ExampleFoodImageDataset");
% 
% if ~exist(imageFolder,'dir') % download only once
%     disp('Downloading Flower Dataset (218 MB)...');
%     websave(filename,url);
%     untar(filename,downloadFolder)
% end

% As image data is available in my computer, I will be using the same data
% here.
%% Load the image data.
impath = 'H:\Research\Datasets (RealNoisy Image)\Deep Learning (NN_CNN) in MATLAB\ExampleFoodImageDataset';
imds = imageDatastore(impath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Counting Number of Levels. and Total Images in Each folder
T = countEachLabel(imds);

% Counting total number of images in the dataset.
imgTot = length(imds.Files);

% Read a single (Randomly selected) image.
im = readimage(imds, 100);
figure,
imshow(im, [])
title('A Single image from Dataset')

% Another-way to find image.
% Find the first instance of an image for each category
hamburger = find(imds.Labels == 'hamburger', 10); % Find 10 hamburger image

figure
imshow(readimage(imds,hamburger(4))) % Display any hamburger image by selecting any number between 1 to 10.

% We will select only 4 classes for image classification.
% We will make appropriate change in the 'imds' variable.
categories = {'french_fries', 'hamburger', 'pizza', 'sushi'};

imds = imageDatastore(fullfile(impath,categories), ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Counting Number of Levels. and Total Images in Each folder
tbl = countEachLabel(imds);

% Counting total number of images in the dataset.
imgTot = length(imds.Files);
% Determine the smallest amount of images in a category
minSetCount = min(tbl{:,2}); 

% We will run the example only for 100 images to limit the trainig process.
maxNumImages = 100;
minSetCount = min(maxNumImages,minSetCount);

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

% --- LOAD the Pre-Trained CNN Model ----
net = resnet50();

% Display the CNN architecture.
figure
plot(net)
title('Architecture of ResNet-50')
set(gca,'YLim',[150 170]);

% Get the input image size of the image from the first layer of ResNet
net.Layers(1)
imageSize = net.Layers(1).InputSize;

% Inspect the last layer. 
net.Layers(end)
numel(net.Layers(end).ClassNames) % 1000 class clasification
% The ResNet was originally proposed for classification of 1000 different
% classes. However we will use it for classification of 4 classes. Note
% ResNet will only be used to get the features at the higher level for SVM
% classification.

% Prepare Training and testing image data for classification.
% 30% Training data and 70%test data
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');

% As the ResNet50 is trained for input size "224*224*3", resize the
% training and testing image data to same size.
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet,...
    'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, ...
    'ColorPreprocessing', 'gray2rgb');

% Extract Features: Initally, we will inspect the weights of first few
% layers [These weights capture blob and edge features]. 
% Get the network weights for the second convolutional layer
w1 = net.Layers(2).Weights; % 7*7*7 

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights')

% ---- We are Interested in the Features extracted from the last layer
% before fully connected layer. ---- This will provide all the fetures that
% could be used by SVM for classification.
featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Train a Multiclass classifier using ResNet50 CNN features.
% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
% Save the classifier for 
save classifier classifier

% Evaluate the classifier using Confusion Matrix
% Extract test features using the CNN
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

% Display the mean accuracy
mean(diag(confMat))

% Apply trained classifier on Test image set.
testImage = readimage(testSet,1);
testLabel = testSet.Labels(1)

figure,
imshow(testImage,[])
title1 = strcat('Image belongs to:', string(testSet.Labels(1)));
title(title1)

