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
outputSize = [640 480];

% Define a input basefolder 
uiwait(msgbox('Please select the Top-Level FaceDataset directory.'));
inputbasefolder = uigetdir(start_path);
if inputbasefolder == 0
	return;
end
fprintf('The top level folder is "%s".\n', inputbasefolder);

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
% In built Face detector - 74% accurate for Yale B Dataset
% FaceDetect = vision.CascadeObjectDetector
% Will load if no face detection training done.

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
