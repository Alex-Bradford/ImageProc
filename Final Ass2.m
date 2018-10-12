%% Crop Images to a Detected Face
%% Damien Smith - 13039957 - Image Processing Ass 2
% In order to run this code you will need the following:
% 1. Download Yale B Face Dataset:
% [Found at: http://vision.ucsd.edu/~iskwak/ExtYaleDatabase/ExtYaleB.html]
% - Original images [2 gb]
% - Cropped images [84mb]
% 2. Create 'successful' and 'failed' output folders
% -optional-
% 3. Create/find/download:
% [For training the face detector model, if not possible it will default to the inbuilt detector]
% - Positive Face images (a folder of images with faces)
% - Negative Face images (a folder of images without faces)
% [provided examples found at: https://drive.google.com/drive/folders/1R294eKzyphAufWrCLrNbiRXBQnAyGK5b?usp=sharing]
% - Faces [2mb]
% - NotFaces [209mb]
%
% You will be prompted to set the directory location of these downloaded folders
%
% You will be required to label faces from the Positive Faces training set
% - After labeling, be sure to 'export' the labels to the workspace calling
% it 'gTruth' (the default name)
% - At this point, a warning message will appear, do not clear this message
% until the gTruth has been exported
% 
% Sit back and enjoy the images process ^_^
%
% ***********************************************************************************
% The Inbuilt face detector successfully identifies - 12080/16380 images (74% accurate)
% It takes approximately 30 minutes to run with the Yale B Face dataset
% Training mode Best model found 96.5% of faces from dataset (using provided positive/negative instances and labeling each face full-frame)

% Define a start_path.
start_path = fullfile(matlabroot, '');
if ~exist(start_path, 'dir')
	start_path = matlabroot;
end

%define output size
outputSize = [192 168];

% Define a input basefolder 
uiwait(msgbox('Please select the Top-Level FaceDataset directory.'));
inputbasefolder = uigetdir(start_path);

blankstring = ' ';
try     imds = imageDatastore(inputbasefolder, ...
    'IncludeSubfolders',true,'LabelSource', 'foldernames');
    catch e %e is an MException struct
        fprintf(1,'There was an error! The message was:\n%s',e.message);
        fprintf(1,'\nPlease try again. \n%s')
        fprintf('\n')
        temp = 1
    return;
end

if inputbasefolder == blankstring | temp == 1
    fprintf('Failed to find dataset, try again.\n')
	return;
else

fprintf('The top level folder is "%s".\n', inputbasefolder);

K = menu('Is the Face Dataset Cropped?','Yes','No')

if K == 1
    imds = imageDatastore(inputbasefolder, ...
    'IncludeSubfolders',true,'LabelSource', 'foldernames');
end

if K == 2
% Define output basefolders:
uiwait(msgbox('Please select the successfully cropped face output directory. *note* this folders contents will be deleted prior to detection'));
outputBaseFolder = uigetdir(start_path);
if outputBaseFolder == 0
	return;
end
fprintf('The successfully cropped face output folder is "%s".\n', outputBaseFolder);

% Define failed basefolder:
uiwait(msgbox('Please select the failed cropped face output directory. *note* this folders contents will be deleted prior to detection'));
failedBaseFolder = uigetdir(start_path);
if failedBaseFolder == 0
	return;
end
fprintf('The failed cropped face output folder is "%s".\n', failedBaseFolder);

% Cleanup - empties the output folders
if ~isempty(outputBaseFolder)
delete([outputBaseFolder filesep '*.*']);
end
if ~isempty(failedBaseFolder)
delete([failedBaseFolder filesep '*.*']);
end

% Specify the file pattern.
% Is set up for '.pgm' files
filePattern = sprintf('%s/**/*.pgm', inputbasefolder);
% Get ALL images
files = dir(filePattern);

% Create logical list of folders.
%isFolder = [files.isdir]; 
% Deletes folders from the list (if they exist)
%if ~isempty(isFolder)
%allFileInfo(isFolder) = [];
%end

% Convert files to a table
fileTable =struct2table(files);
% Add name column to table
fileTable.name=string(fileTable.name);
% Add folder column to table
fileTable.folder=string(fileTable.folder);
% Add full file name column to table
fileTable.fullFileName=fileTable.folder + filesep+ fileTable.name;

A = menu('Would you like to train your own Face Detector Model?','Yes','No')

if A == 1
    imds = imageDatastore(inputbasefolder, ...
    'IncludeSubfolders',true,'LabelSource', 'foldernames');

%% Train an Object Detector - for Faces
% Load positive samples.
uiwait(msgbox('Please select folder containing positive face samples for model training. Select Faces and export gTruth to workspace. If this fails, the default face detection will be used'));
tempfolder = uigetdir(start_path);
if isempty(tempfolder)
	FaceDetect = vision.CascadeObjectDetector
else
fprintf('The positive face samples folder is "%s".\n', tempfolder);
imageLabeler(tempfolder)
mydlg = warndlg('Close this dialogue AFTER you have exported your gTruth from image Labeler', 'Warning');
waitfor(mydlg);
% convert gTruth into positiveInstances table
temp = gTruth.DataSource.Source(:,1:1);
%temp = cell2table(temp)
temp2 = gTruth.LabelData(:,1:1);
%temp2 = cell2table(temp2)
positiveInstances = [temp temp2];

%Specify the folder for negative images.
uiwait(msgbox('Please select folder containing negative face samples for model training.'));
negativeFolder = uigetdir(start_path);
if tempfolder == 0
	FaceDetect = vision.CascadeObjectDetector
end

%Create an imageDatastore object containing negative images.
negativeImages = imageDatastore(negativeFolder);

%Train a cascade object detector called 'Faces.xml' using HOG features. NOTE: The command can take several minutes to run.
trainCascadeObjectDetector('model.xml',positiveInstances,negativeFolder,'FalseAlarmRate',0.1,'NumCascadeStages',5);

% Store the model
FaceDetect = vision.CascadeObjectDetector('model.xml');
end
end

if A == 2
    FaceDetect = vision.CascadeObjectDetector
end


%% Read input image and detect face
for i=1:height(fileTable)
       fileName=fileTable.fullFileName{i};
       fprintf('Processing image (%d of %d): %s...', i,height(fileTable),fileName);
       
       % Read Image
       I = imread(fileName);
       
       % Perform Detection
       BB = step(FaceDetect, I);
              
       if ~isempty(BB)
                  BB = BB(1:1,1:4);
       % Extract Face Area
       x=BB(1);
       y=BB(2);
       w=BB(3);
       h=BB(4);

       detectedArea=I(y:y+h,x:x+w,:);
          
       Resize = imresize(detectedArea, outputSize);
       % Output image of detected Face Area
       outputFileName=[outputBaseFolder filesep 'image_' num2str(i) '.pgm'];
       imwrite(Resize, outputFileName, 'pgm')

       fprintf('Done. \n');
       else
           copyfile(fileName, failedBaseFolder);
           fprintf('Failed to Detect Face.\n');
        end
end

filePattern = sprintf('%s/**/*.pgm', outputBaseFolder);
% Get ALL images
files = dir(filePattern);

% Convert files to a table
positiveFileTable =struct2table(files);
% Add name column to table
positiveFileTable.name=string(positiveFileTable.name);
% Add folder column to table
positiveFileTable.folder=string(positiveFileTable.folder);
% Add full file name column to table
positiveFileTable.fullFileName=positiveFileTable.folder + filesep+ positiveFileTable.name;

successrate = (height(positiveFileTable)/height(fileTable)) *100
% successrate = round(successrate * 100)/100
successrate = uint8(successrate)

fprintf('Successully cropped (%d of %d) images: [successrate = %d%%] \n', height(positiveFileTable), height(fileTable), successrate);

imds = imageDatastore(outputBaseFolder, ...
    'IncludeSubfolders',true,'LabelSource', 'foldernames');

for_comparison = imread(fileTable.fullFileName{1});
    figure
    imshowpair(for_comparison,imread(imds.Files{1}),'montage')
    title('Here is an example of the cropping you just did!');
end

M = menu('How would you like to Pre-Process the data?','Histogram Equalization','Salt & Pepper Noise','Gaussian Blur', 'Gaussian Filter', 'Median Filter', 'High-Pass Filter', 'No Pre-Processing')
if M == 1
% Run histogram equalization
    % Calculate the number of files in the data store
    for_comparison = imread(imds.Files{1});
    label_counts = imds.countEachLabel;
    no_of_files = 0;
    for i=1:height(label_counts)
        no_of_files = no_of_files + label_counts{i,2};
    end
    % Apply histogram equalisation to each image
    for i=1:no_of_files
        img = readimage(imds,i);
        img = histeq(img,10);
        imwrite(img,imds.Files{i}); % push new img to its original location
    end
    % read the images again, they have changed.
    if K == 1
        imds = imageDatastore(inputbasefolder, ...
        'IncludeSubfolders',true,'LabelSource', 'foldernames');
    end
    if K == 2
        imds = imageDatastore(outputBaseFolder, ...
        'IncludeSubfolders',true,'LabelSource', 'foldernames');
    end
    figure
    imshowpair(for_comparison,imread(imds.Files{1}),'montage')
    title('Here is an example of the pre processing you just did!');
    
    

end

if M == 2
% Salt & Pepper
% Calculate the number of files in the data store
    for_comparison = imread(imds.Files{1});
    label_counts = imds.countEachLabel;
    no_of_files = 0;
    for i=1:height(label_counts)
        no_of_files = no_of_files + label_counts{i,2};
    end
    % Add Salt&Pepper Noise to each image
    for i=1:no_of_files
        img = readimage(imds,i);
        img = imnoise(img,'salt & pepper',0.02);
        imwrite(img,imds.Files{i}); % push new img to its original location
    end
    % read the images again, they have changed.
    if K == 1
        imds = imageDatastore(inputbasefolder, ...
        'IncludeSubfolders',true,'LabelSource', 'foldernames');
    end
    if K == 2
        imds = imageDatastore(outputBaseFolder, ...
        'IncludeSubfolders',true,'LabelSource', 'foldernames');
    end
    figure
    imshowpair(for_comparison,imread(imds.Files{1}),'montage')
    title('Here is an example of the pre processing you just did!');
end

if M == 3
% Gaussian Blur
    for_comparison = imread(imds.Files{1});
    label_counts = imds.countEachLabel;
    no_of_files = 0;
    for i=1:height(label_counts)
        no_of_files = no_of_files + label_counts{i,2};
    end
    % Add Gaussian Noise to each image
    for i=1:no_of_files
        img = readimage(imds,i);
        img = imnoise(img,'gaussian',0,0.025);
        imwrite(img,imds.Files{i}); % push new img to its original location
    end
    % read the images again, they have changed.
    if K == 1
        imds = imageDatastore(inputbasefolder, ...
        'IncludeSubfolders',true,'LabelSource', 'foldernames');
    end
    if K == 2
        imds = imageDatastore(outputBaseFolder, ...
        'IncludeSubfolders',true,'LabelSource', 'foldernames');
    end
    figure
    imshowpair(for_comparison,imread(imds.Files{1}),'montage')
    title('Here is an example of the pre processing you just did!');
end

if M == 4
    %Gaussian Filter%
 for_comparison = imread(imds.Files{1});
    label_counts = imds.countEachLabel;
    no_of_files = 0;
    for i=1:height(label_counts)
        no_of_files = no_of_files + label_counts{i,2};
    end
    % Apply Gaussian Filter to each image
    for i=1:no_of_files
        img = readimage(imds,i);
        img = imgaussfilt(img);
        imwrite(img,imds.Files{i}); % push new img to its original location
    end
    % read the images again, they have changed.
    if K == 1
        imds = imageDatastore(inputbasefolder, ...
        'IncludeSubfolders',true,'LabelSource', 'foldernames');
    end
    if K == 2
        imds = imageDatastore(outputBaseFolder, ...
        'IncludeSubfolders',true,'LabelSource', 'foldernames');
    end
    figure
    imshowpair(for_comparison,imread(imds.Files{1}),'montage')
    title('Here is an example of the pre processing you just did!');
end
if M == 5
    %Median Filter%
     for_comparison = imread(imds.Files{1});
    label_counts = imds.countEachLabel;
    no_of_files = 0;
    for i=1:height(label_counts)
        no_of_files = no_of_files + label_counts{i,2};
    end
    % Apply Median Filter to each image
    for i=1:no_of_files
        img = readimage(imds,i);
        img = medfilt2(img);
        imwrite(img,imds.Files{i}); % push new img to its original location
    end
    % read the images again, they have changed.
    if K == 1
        imds = imageDatastore(inputbasefolder, ...
        'IncludeSubfolders',true,'LabelSource', 'foldernames');
    end
    if K == 2
        imds = imageDatastore(outputBaseFolder, ...
        'IncludeSubfolders',true,'LabelSource', 'foldernames');
    end
    figure
    imshowpair(for_comparison,imread(imds.Files{1}),'montage')
    title('Here is an example of the pre processing you just did!');
end
if M == 6
    %High-Pass Filter%
     for_comparison = imread(imds.Files{1});
    label_counts = imds.countEachLabel;
    no_of_files = 0;
    for i=1:height(label_counts)
        no_of_files = no_of_files + label_counts{i,2};
    end
    % Apply High-Pass Filter to each image
    for i=1:no_of_files
        img = readimage(imds,i);
        % Perform Detection
        kernel = [ -1 -1 -1; -1 8 -1; -1 -1 -1];
        img = imfilter(img, kernel, 'same');
        imwrite(img,imds.Files{i}); % push new img to its original location
    end
    % read the images again, they have changed.
    if K == 1
        imds = imageDatastore(inputbasefolder, ...
        'IncludeSubfolders',true,'LabelSource', 'foldernames');
    end
    if K == 2
        imds = imageDatastore(outputBaseFolder, ...
        'IncludeSubfolders',true,'LabelSource', 'foldernames');
    end
    figure
    imshowpair(for_comparison,imread(imds.Files{1}),'montage')
    title('Here is an example of the pre processing you just did!');    
end
if M == 7
% DO NOTHING
% Output to 'processing folder'
end

N = menu('How would you like to extract your features and classify the faces','SVM with HoG','SVM with SURF','Convolution Neural Network', 'Naive Bayes with ??', 'Lucky Dip?')

if N == 1
% SVM with HOG
    % resize all images
    imds.ReadSize = numpartitions(imds);
    imds.ReadFcn = @(loc)imresize(imread(loc), [192, 168]);
    % split into train and test
    uniq_labels = countEachLabel(imds);
    uniq_labels = height(uniq_labels);
    label_counts = imds.countEachLabel;
    numTrainFiles = round(0.8*label_counts{1,2},0);
    [imdsTraining, imdsTest] = splitEachLabel(imds, numTrainFiles, 'randomize');
    % create array to store features
    trainingFeatures = zeros(numTrainFiles*uniq_labels,16560);
    % calc number of files
    tr_no_of_files = 0;
    tr_label_counts = imdsTraining.countEachLabel;
    for i=1:height(tr_label_counts);
        tr_no_of_files = tr_no_of_files + tr_label_counts{i,2};
    end
    % Get HOGFeatures for each image
    featureCount = 1;
    for i=1:tr_no_of_files;
        trainingFeatures(featureCount,:) = extractHOGFeatures(readimage(imdsTraining,i));
        trainingLabel{featureCount} = string(imdsTraining.Labels(i));
        featureCount = featureCount + 1;
        %personIndex{i} = training(i).Description;
    end
    trainingLabel = cellstr(trainingLabel);
    % train the model
    model = fitcecoc(trainingFeatures, trainingLabel);
    % test the model and calc accuracy
    % calc number of files
    te_no_of_files = 0;
    te_label_counts = imdsTest.countEachLabel;
    for i=1:height(te_label_counts)
        te_no_of_files = te_no_of_files + te_label_counts{i,2};
    end
    % for each test file
    correct_pred = 0;
    false_pred = 0;
    for i=1:te_no_of_files
        pred_label = predict(model,extractHOGFeatures(readimage(imdsTest,i)));
        real_label = cellstr(string(imdsTest.Labels(i)));
        if pred_label{1} == real_label{1}
            correct_pred = correct_pred + 1;
        else
            false_pred = false_pred + 1;
        end
    end
    accur = correct_pred/(correct_pred+false_pred)
% Output accuracy
end

if N == 2
% SVM with SURF
% Output accuracy
 
% Split the data into a training and a test set
[trainingSet, testSet] = splitEachLabel(imds, 0.75, 'randomize');
% Extract SURF Features and store them into a bag of words
yaleBagOfWords = bagOfFeatures(trainingSet);
% Train the classifier using the training set and bag of words
yaleClassifier = trainImageCategoryClassifier(trainingSet, yaleBagOfWords);
% Evaluate the classifier using the training set
trainingConfMatrix = evaluate(yaleClassifier, trainingSet)
mean(diag(trainingConfMatrix))
% Evaluate the classifier using the test set 
testConfMatrix = evaluate(yaleClassifier, testSet)
mean(diag(testConfMatrix))
 
end

if N == 3
% Convolution Neural Network
% Output accuracy

% Resize the images into a common size
imds.ReadSize = numpartitions(imds);
imds.ReadFcn = @(loc)imresize(imread(loc), [192, 168]);
% Split the data into a training and a test set
[trainingSet, testSet] = splitEachLabel(imds, 0.75, 'randomize');
% Obtain the number of classes that are in the face database
numberOfClasses = size(countEachLabel(imds), 1);
% Create the archtiecture of the network / Define the number of layers

layers = [
    % Specficy the size of the image (192 x 168). 1 means it is B/W
    imageInputLayer([192 168, 1])
    
    % Create the first convolutional layer, where the first 8 is the filter size (8
    % x 8 filter) and the second 8 is the number of neurons that connect to
    % the same region of the image
    convolution2dLayer(8,8,'Padding','same')
    % Normalise the activations and gradients to speed up the network
    % training time and to reduce network sensitivity
    batchNormalizationLayer
    % A non-linear activation function. Should be followed by a batch
    % normalisation layer
    reluLayer
    
    % Down sample the feature map to remove reundant information. Allows
    % more filters to be used without increasing the amount of computation
    % required per layer. 
    maxPooling2dLayer(2,'Stride',2)
    
    % Add another convolutional layerm but with a larger filter this time
    convolution2dLayer(8,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2)
       
    % Add another convolutional layer but with a larger filter this time
    convolution2dLayer(8,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(numberOfClasses)
    softmaxLayer
    classificationLayer];

% Define the options for the network

options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'auto', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',9, ...
    'Shuffle','every-epoch', ...
    'ValidationData',testSet, ...
    'ValidationFrequency',40, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train the network
net = trainNetwork(trainingSet, layers, options);

% Calculate the accuracy of the network using the test set
YPred = classify(net, testSet);

% Testing time
sum(YPred == testSet.Labels) / numel(YPred)

end

if N == 4
% Naives Bayes with ??
% Output accuracy
end

if N == 5
% RANDOM!!
% Output accuracy
end
end
% Using the model just created, create a dialogue box to select label and
% image and predict on that image.
