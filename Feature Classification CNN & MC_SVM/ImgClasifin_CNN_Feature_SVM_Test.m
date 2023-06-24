clear 
close all
clc

% This program is used for testing the classifier accuracy with user given
% image data.
load classifier
net = resnet50();
layers = net.Layers;
imageSize = net.Layers(1).InputSize;
featureLayer = 'fc1000';

% Select an image file for testing
[filename, pathname] = uigetfile('*.*', 'Pick an image File');

if isequal(filename,0) || isequal(pathname,0)
    disp('user select cancel')
else
    ImgFilename = strcat(pathname, filename);
    img = imread(ImgFilename);
    ds = augmentedImageDatastore(imageSize, img, 'ColorPreprocessing', 'gray2rgb');

    FeaturesTrain = activations(net, ds, featureLayer, 'OutputAs', 'columns');
    category = predict(classifier, FeaturesTrain, 'ObservationsIn', 'columns');
end

figure,
imshow(img, [])
title1 = strcat('Image Category: ', string(category));
title(title1)
