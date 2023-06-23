% ------ Image Classification using Pre-Trained CNN in MATLAB ------
% ---------- googleNet and ResNet101 --------------------

clear
close all
clc
% Load the pretrained network
net = googlenet;

% Read an image for classification. 
I = imread("peacock.jpg");
inputSize = net.Layers(1).InputSize;
I = imresize(I,inputSize(1:2));

% 1000 Different classes are in googleNet. Let see randomly 20 different
% classes.
classNames = net.Layers(end).ClassNames; %Get the Number of classes from last layer
numClasses = numel(classNames); % Total classes prsent.
disp(classNames(randperm(numClasses,20))) % Randomly choose 20 classes and display.

% Classify the data:
label = classify(net,I);
figure
imshow(I)
title(string(label))

% Find the percentage of accuracy for given image I:
[label,scores] = classify(net,I);

% Display Top Prediction: We have Considered Top 10-----
[~,idx] = sort(scores,'descend');
idx = idx(1:10);
classNamesTop = net.Layers(end).ClassNames(idx);
scoresTop = scores(idx);

figure
barh(scoresTop)
xlim([0 1])
title('Top 10 Predictions')
xlabel('Probability')
yticklabels(classNamesTop)

