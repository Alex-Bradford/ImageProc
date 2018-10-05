%% Crop images to a Detected Face
%% Damien Smith - 13039957 - Image Processing Ass 2
% In order to run this code download the Yale B Face Dataset at the
% following url: http://vision.ucsd.edu/~iskwak/ExtYaleDatabase/ExtYaleB.html
% be sure to download the 'original images' - 2 Gb download.
% successfully identifies - 12080 images
% fails to identify - 4300 images
% Takes approximately 45 minutes to run with the Yale B Face dataset

% Define a input basefolder.
start_path = fullfile(matlabroot, 'C:\Users\Damien\Documents\Uni\Image Processing and Pattern Recognition\Assignment\Yale FaceDatabase\NotCropped\ExtendedYaleB');
if ~exist(start_path, 'dir')
	start_path = matlabroot;
end
% User selects where Face Dataset is located 
uiwait(msgbox('Please select the Top-Level FaceDataset directory.'));
inputbasefolder = uigetdir(start_path);
if inputbasefolder == 0
	return;
end
fprintf('The top level folder is "%s".\n', inputbasefolder);

%define output size
outputSize = [640 480];

% Create output basefolders:
outputBaseFolder='C:\Users\Damien\Documents\Uni\Image Processing and Pattern Recognition\Assignment\Yale FaceDatabase\NotCropped\ExtendedYaleB\processed'
failedBaseFolder='C:\Users\Damien\Documents\Uni\Image Processing and Pattern Recognition\Assignment\Yale FaceDatabase\NotCropped\ExtendedYaleB\failed'

% Cleanup - empties the output folders
if ~isempty(outputBaseFolder)
delete([outputBaseFolder filesep '*.*']);
end
if ~isempty(failedBaseFolder)
delete([failedBaseFolder filesep '*.*']);
end

% Specify the file pattern.
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

%% Store an Object Detector - for Faces
% From the computer vision System toolbox, The cascade object detector uses 
% the Viola-Jones algorithm to detect people’s faces (without training)
FaceDetect = vision.CascadeObjectDetector;

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