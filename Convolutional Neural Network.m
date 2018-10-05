%% Load the image data
digitDatasetPath = fullfile('CroppedYale');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource', 'foldernames');

%% Count the number of images in each category

labelCount = countEachLabel(imds)

%% Calculate the size of each image

img = readimage(imds, 1)
[height, width] = size(img)

%% Image Resizing
imds.ReadSize = numpartitions(imds);
imds.ReadFcn = @(loc)imresize(imread(loc), [192, 168]);

%% Split the images into a training and validation set

numTrainFiles = 49
[imdsTrain, imdsValidation] = splitEachLabel(imds, numTrainFiles, 'randomize');


%% Extract the number of classes

numberOfClasses = size(countEachLabel(imds), 1)

%% Define the architecture of the network

layers = [
    imageInputLayer([192 168, 1])
    
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
    
    fullyConnectedLayer(numberOfClasses)
    softmaxLayer
    classificationLayer];


%% Define the options for the network

options = trainingOptions('sgdm', ...
    'ExecutionEnvironment', 'auto', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',8, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',40, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Train the network

net = trainNetwork(imdsTrain, layers, options);

%% Calculate accuracy 
YPred = classify(net, imdsValidation)

%% Testing time
sum(YPred == imdsValidation.Labels) / numel(YPred)




