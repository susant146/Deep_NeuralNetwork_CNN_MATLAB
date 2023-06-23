% ------ Read and display multiple images in MATLAB --------
% storing all the images from external dataset in MATALB will not always be
% fisible; The follwoing command is used to store the data; In future these
% data can be used for training and testing for Deep CNN models. 
% ------------------ imageDatastore -------------
% Download Dataset:
% https://www.mathworks.com/supportfiles/nnet/data/ExampleFoodImageDataset.zip
% The Example Food 
% Images data set contains 978 photographs of food in nine classes (_ceaser_salad_, 
% _caprese_salad_, _french_fries_, _greek_salad_, _hamburger_, _hot_dog_, _pizza_, 
% _sashimi_, and _sushi_).
% 
% Other dataset is available at <https://jp.mathworks.com/help/deeplearning/ug/data-sets-for-deep-learning.html. 
% https://jp.mathworks.com/help/deeplearning/ug/data-sets-for-deep-learning.html.> 

clear 
close all
clc

% If dataset not available then use the following command to download and
% unzip the data.
%% ------------ Download and Unzip the Data ----------
% url = "https://www.mathworks.com/supportfiles/nnet/data/ExampleFoodImageDataset.zip";
% downloadFolder = pwd;
% filename = fullfile(downloadFolder,'ExampleFoodImageDataset.zip');
% 
% dataFolder = fullfile(downloadFolder, "ExampleFoodImageDataset");
% if ~exist('ExampleFoodImageDataset.zip')
%     fprintf("Downloading Example Food Image data set (77 MB)... ")
%     websave(filename,url);
%     unzip(filename,downloadFolder);
%     fprintf("Done.\n")
% end

%% If Image Dataset is already available in local folder. Ignore 
% downloading and follow these step.
impath = 'H:\Research\Datasets (RealNoisy Image)\Deep Learning (NN_CNN) in MATLAB\ExampleFoodImageDataset';
imds = imageDatastore(impath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Counting Number of Levels. and Total Images in Each folder
T = countEachLabel(imds);

% Counting total number of images in the dataset.
imgTot = length(imds.Files);

% Read a single (Randomly selected) image.
im = readimage(imds, 200);
figure,
imshow(im, [])
title('A Single image from Dataset')
