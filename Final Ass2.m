%% Image Processing - 31256
% Assignment 2 - Face Detection & Face Recognition
% 
% Please refer to the readme file for instruction on running this code
% 

% Define a start_path.
start_path = fullfile(matlabroot, '');
if ~exist(start_path, 'dir')
	start_path = matlabroot;
end

%define output size
outputSize = [192 168];


fprintf('Please select the Top-Level FaceDataset directory.\n')
fprintf('Click OK and select from the browser.\n')
fprintf('\n')

% Define a input basefolder 
uiwait(msgbox('Please select the Top-Level FaceDataset directory.'));
inputbasefolder = uigetdir(start_path);
blankstring = ' ';
temp = 0;
imds = 0;
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

fprintf('\n')
fprintf('Is the Face Dataset Cropped?.\n')
fprintf('Select Yes or No from the menu.\n')
fprintf('\n')

K = 55;
K = menu('Is the Face Dataset Cropped?','Yes','No')

if K == 0
    fprintf('Menu closed. "%s".\n')
    fprintf('Proceeding to Pre-Processing Menu. "%s".\n')
    fprintf('\n')
end
    
if K == 1
    imds = imageDatastore(inputbasefolder, ...
    'IncludeSubfolders',true,'LabelSource', 'foldernames');
end

if K == 2
    fprintf('Please select the successfully cropped face output directory.\n')
    fprintf('*note* this folders contents will be deleted prior to detection.\n')
    fprintf('Click OK and select from the browser.\n')
    fprintf('\n')
  
% Define output basefolders:
uiwait(msgbox('Please select the successfully cropped face output directory. *note* this folders contents will be deleted prior to detection'));
outputBaseFolder = uigetdir(start_path);
if outputBaseFolder == 0
	fprintf('Failed to define the successful cropped face output folder.\n')
    fprintf('Unable to train model.\n')
    fprintf('Proceeding to Pre-Processing.\n')
    else
fprintf('The successfully cropped face output folder is "%s".\n', outputBaseFolder);
        fprintf('\n')

    fprintf('Please select the failed cropped face output directory.\n')
    fprintf('*note* this folders contents will be deleted prior to detection.\n')
    fprintf('Click OK and select from the browser.\n')
    fprintf('\n')
% Define failed basefolder:
uiwait(msgbox('Please select the failed cropped face output directory. *note* this folders contents will be deleted prior to detection'));
failedBaseFolder = uigetdir(start_path);
if failedBaseFolder == 0
	fprintf('Failed to define the failed cropped face output folder.\n')
    fprintf('Unable to train model.\n')
    fprintf('Proceeding to Pre-Processing.\n')
 else

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

% Convert files to a table
fileTable =struct2table(files);
% Add name column to table
fileTable.name=string(fileTable.name);
% Add folder column to table
fileTable.folder=string(fileTable.folder);
% Add full file name column to table
fileTable.fullFileName=fileTable.folder + filesep+ fileTable.name;

fprintf('\n')
fprintf('Would you like to train your own Face Detector Model?.\n')
fprintf('Select Yes or No from the menu.\n')
fprintf('*note* Selecting No (or closing the menu) will select the default MATLAB face detection model.\n')
fprintf('\n')

A = menu('Would you like to train your own Face Detector Model?','Yes','No')
FaceDetect = vision.CascadeObjectDetector

if A == 1
    imds = imageDatastore(inputbasefolder, ...
    'IncludeSubfolders',true,'LabelSource', 'foldernames');

%% Train an Object Detector - for Faces
gTruth = 0;

fprintf('Please select folder containing positive face samples for model training.\n')
fprintf('The Image Labeller will open.\n')
fprintf('Label as many faces as you would like (the more the better).\n')
fprintf('If you select an empty folder, it is possible to load additional folders from the Image Labeller menu.\n')
fprintf('Use the export gTruth to workspace option.\n')
fprintf('Be sure to save as the default name - gtruth.\n')
fprintf('\n')

fprintf('*NOTE* DO NOT CLOSE THE WARNING until after exporting gtruth.\n')
fprintf('The default model will be used in the event of an error.\n')
fprintf('\n')
% Load positive samples.
uiwait(msgbox('Please select folder containing positive face samples for model training. Select Faces and export gTruth to workspace. If this fails, the default face detection will be used'));
tempfolder = uigetdir(start_path);
if tempfolder == 0
	fprintf('Failed to find positive face samples for model Training.\n')
    fprintf('Training on the default Face Detector Model.\n')
    FaceDetect = vision.CascadeObjectDetector
else
fprintf('The positive face samples folder is "%s".\n', tempfolder);
imageLabeler(tempfolder)
mydlg = warndlg('Close this dialogue AFTER you have exported your gTruth from image Labeler', 'Warning');
waitfor(mydlg);
if gTruth == 0
    fprintf('Failed to find gTruth (Ground Truth).\n')
    fprintf('Training on the default Face Detector Model.\n')
    mydlg = warndlg('Failed to find gTruth (Ground Truth). The Default Face Detector Model will be used.', 'Warning');
    waitfor(mydlg);
    FaceDetect = vision.CascadeObjectDetector
else
% convert gTruth into positiveInstances table
temp = gTruth.DataSource.Source(:,1:1);
%temp = cell2table(temp)
temp2 = gTruth.LabelData(:,1:1);
%temp2 = cell2table(temp2)
positiveInstances = [temp temp2];

fprintf('Please select folder containing negative face samples for model training.\n')
fprintf('\n')
fprintf('The default model will be used in the event of an error.\n')
fprintf('\n')

%Specify the folder for negative images.
uiwait(msgbox('Please select folder containing negative face samples for model training.'));
negativeFolder = uigetdir(start_path);
temp = 0;
try     imds = imageDatastore(inputbasefolder, ...
    'IncludeSubfolders',true,'LabelSource', 'foldernames');
    catch e %e is an MException struct
        fprintf(1,'There was an error! The message was:\n%s',e.message);
        fprintf('\n')
        temp = 1
    return;
end

if negativeFolder == 0
	fprintf('Failed to find negative face samples for model Training.\n')
    fprintf('Training on the default Face Detector Model.\n')
    mydlg = warndlg('Failed to find gTruth (Ground Truth). The Default Face Detector Model will be used.', 'Warning');
    waitfor(mydlg);
    FaceDetect = vision.CascadeObjectDetector
elseif temp == 1
    fprintf('Failed to find negative face samples for model Training.\n')
    fprintf('Training on the default Face Detector Model.\n')
    mydlg = warndlg('Failed to find gTruth (Ground Truth). The Default Face Detector Model will be used.', 'Warning');
    waitfor(mydlg);
    FaceDetect = vision.CascadeObjectDetector
else

%Create an imageDatastore object containing negative images.
negativeImages = imageDatastore(negativeFolder);

%Train a cascade object detector called 'Faces.xml' using HOG features. NOTE: The command can take several minutes to run.
trainCascadeObjectDetector('model.xml',positiveInstances,negativeFolder,'FalseAlarmRate',0.1,'NumCascadeStages',5);

% Store the model
FaceDetect = vision.CascadeObjectDetector('model.xml');
end
end
end
end

if A == 2 | A == 0
    FaceDetect = vision.CascadeObjectDetector
    fprintf('Training on the default Face Detector Model.\n')
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
    
img = imread(fileTable.fullFileName{1});
detectedImg = insertObjectAnnotation(img ,'rectangle',BB(1,:),'Detected Face');
figure,
imshow(detectedImg)
title('Bounding box of detected Face')
    
end
end

if imds == 0
        imds = imageDatastore(inputbasefolder, ...
    'IncludeSubfolders',true,'LabelSource', 'foldernames');
end


end

fprintf('How would you like to Pre-Process the data? \n')
fprintf('Select from the menu. \n')

M = menu('How would you like to Pre-Process the data?','Histogram Equalization','Salt & Pepper Noise','Gaussian Blur', 'Gaussian Filter', 'Median Filter', 'High-Pass Filter', 'No Pre-Processing')
if M == 1
% Run histogram equalization
    % Calculate the number of files in the data store
    disp('Pre processing now...')
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
    disp('Pre processing now...')
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
    disp('Pre processing now...')
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
    disp('Pre processing now...')
    for_comparison = imread(imds.Files{1});
    label_counts = imds.countEachLabel;
    no_of_files = 0;
    for i=1:height(label_counts)
        no_of_files = no_of_files + label_counts{i,2};
    end
    % Apply Gaussian Filter to each image
    for i=1:no_of_files
        img = readimage(imds,i);
        img = imgaussfilt(img,2);
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
    disp('Pre processing now...')
    for_comparison = imread(imds.Files{1});
    label_counts = imds.countEachLabel;
    no_of_files = 0;
    for i=1:height(label_counts)
        no_of_files = no_of_files + label_counts{i,2};
    end
    % Apply Median Filter to each image
    for i=1:no_of_files
        img = readimage(imds,i);
        img = medfilt2(img,[7 7]);
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
    disp('Pre processing now...')
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

N = menu('How would you like to extract your features and classify the faces','SVM with HOG','SVM with SURF','Convolution Neural Network', 'Decision Tree with HOG', 'Naive Bayes with LBP','Lucky Dip?')

if N == 1
% SVM with HOG
    disp('Extracting features and training the model...')
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
    disp('Testing the model...')
    correct_pred = 0;
    false_pred = 0;
    for i=1:te_no_of_files
        pred_label = predict(model,extractHOGFeatures(readimage(imdsTest,i)));
        real_label = cellstr(string(imdsTest.Labels(i)));
        array_real_labels{i} = real_label;
        array_pred_labels{i} = pred_label;
        if pred_label{1} == real_label{1}
            correct_pred = correct_pred + 1;
        else
            false_pred = false_pred + 1;
        end
    end
    disp(['The accuracy of the model was... '])
    accuracy = correct_pred/(correct_pred+false_pred)
    msgbox('Maximise the confusion matrix to enhance readability')
    B = categorical(string(array_real_labels));
    C = categorical(string(array_pred_labels));
    figure(1);
    title('Maximise the window');
    plotconfusion(B,C);
    %fontsize
    set(findobj(gca,'type','text'),'fontsize',7);
% Output accuracy
end

if N == 2
% SVM with SURF
% Output accuracy
disp('Extracting features and training the model...')
% Split the data into a training and a test set
[trainingSet, testSet] = splitEachLabel(imds, 0.75, 'randomize');
% Extract SURF Features and store them into a bag of words
yaleBagOfWords = bagOfFeatures(trainingSet);
% Train the classifier using the training set and bag of words
yaleClassifier = trainImageCategoryClassifier(trainingSet, yaleBagOfWords);
% Evaluate the classifier using the training set
disp('Testing the model...')
trainingConfMatrix = evaluate(yaleClassifier, trainingSet)
mean(diag(trainingConfMatrix))
% Evaluate the classifier using the test set 
disp('The accuracy of the model was...')
testConfMatrix = evaluate(yaleClassifier, testSet)
mean(diag(testConfMatrix))
 
end

if N == 3
% Convolution Neural Network
% Output accuracy
disp('Extracting features and training the model...')
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
disp('Testing the model...')
YPred = classify(net, testSet);

% Testing time
disp('The accuracy of the model was...')
sum(YPred == testSet.Labels) / numel(YPred)

end

if N == 4

% Decision Tree with HOG
    disp('Extracting features and training the model...')
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
    model = fitctree(trainingFeatures, trainingLabel);
    % test the model and calc accuracy
    % calc number of files
    te_no_of_files = 0;
    te_label_counts = imdsTest.countEachLabel;
    for i=1:height(te_label_counts)
        te_no_of_files = te_no_of_files + te_label_counts{i,2};
    end
    % for each test file
    disp('Testing the model...')
    correct_pred = 0;
    false_pred = 0;
    for i=1:te_no_of_files
        pred_label = predict(model,extractHOGFeatures(readimage(imdsTest,i)));
        real_label = cellstr(string(imdsTest.Labels(i)));
        array_real_labels{i} = real_label;
        array_pred_labels{i} = pred_label;
        if pred_label{1} == real_label{1}
            correct_pred = correct_pred + 1;
        else
            false_pred = false_pred + 1;
        end
    end
    disp(['The accuracy of the model was... '])
    accuracy = correct_pred/(correct_pred+false_pred)
    msgbox('Maximise the confusion matrix to enhance readability')
    B = categorical(string(array_real_labels));
    C = categorical(string(array_pred_labels));
    figure(1);
    title('Maximise the window');
    plotconfusion(B,C);
    %fontsize
    set(findobj(gca,'type','text'),'fontsize',7);
% Output accuracy
end

if N == 5
% Naive Bayes with Local Binary Patterns
nbImgDS = imds;

%split Set into training and testing
[trainingSet, testSet] = splitEachLabel(nbImgDS, 0.75, 'randomize');

%Find number of training values
numberOfTrainingValues = 0;
for i =1:height(countEachLabel(trainingSet))
    numberOfTrainingValues = numberOfTrainingValues + trainingSet.countEachLabel{i,2};
end

%Create an zero filled matrix. 59 is because that is the number of
%dimensions LBP will output
trainingFeatures = zeros(numberOfTrainingValues, 59);

% Get features of each photo and store it in a matrix
featureCount = 1;
for i=1:numberOfTrainingValues
	trainImage = readimage(trainingSet, i);
	trainImage = imresize(trainImage, [192 168]);
	trainingFeatures(featureCount, :) = extractLBPFeatures(trainImage, 'Upright',true);
	trainingLabel{featureCount} = string(trainingSet.Labels(i));
	featureCount = featureCount + 1;
end
trainingLabel = cellstr(trainingLabel);

% Create the Model
nbClassifier = fitcnb (trainingFeatures, trainingLabel);

correct = 0;
incorrect = 0;
numberOfTestValues = 0;

%Get Number of Test Values
for i =1:height(countEachLabel(testSet))
    numberOfTestValues = numberOfTestValues + testSet.countEachLabel{i,2};
end

%Check and Output Accuracy
for j=1:numberOfTestValues
	testImage = readimage(testSet, j);
	testImage = imresize(testImage, [192 168]);
	testFeatures =  extractLBPFeatures(testImage, 'Upright',true);
	predictLabel = predict(nbClassifier, testFeatures);
	realLabel = cellstr(string(testSet.Labels(j)));
	if  predictLabel{1} == realLabel{1}
		correct = correct + 1;
	else
		incorrect = incorrect + 1;
	end
end

accuracy = correct/(correct + incorrect)

end

P = menu('Would you like to make a prediction on a random image to try it out?','Yes','No')
if P == 1
    randomLabelIndex = randi([1 uniq_labels],1);
    figure
    imshow(imread(imdsTest.Files{randomLabelIndex}));
    title(['Here is a random picture, it has a class label of ' string(imdsTest.Labels(randomLabelIndex))]);
    pred_label = predict(model,extractHOGFeatures(readimage(imdsTest,randomLabelIndex)));
    z = 0
    while z == 0
        for i=1:te_no_of_files
            if string(imdsTest.Labels(i)) == string(pred_label)
                z = 1
                img = imread(imdsTest.Files{i});
            end
        end
    end
    Q = menu('Would you like to see the predicted image?','Yes','No')
    if Q == 1
        figure
        imshow(img)
        title(['The predicted class label is... ' string(pred_label)]);
    end
end
end
