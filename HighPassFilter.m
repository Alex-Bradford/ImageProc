%High Pass Filters%

start_path = fullfile(matlabroot, '');
if ~exist(start_path, 'dir')
	start_path = matlabroot;
end

% Define a input basefolder 
uiwait(msgbox('Please select the Top-Level FaceDataset directory.'));
inputbasefolder = uigetdir(start_path);
if inputbasefolder == 0
	return;
end
fprintf('The top level folder is "%s".\n', inputbasefolder);

% Define output basefolders:
uiwait(msgbox('Please select the successfully high passed filted face output directory. *note* this folders contents will be deleted prior to detection'));
outputBaseFolder = uigetdir(start_path);
if outputBaseFolder == 0
	return;
end
fprintf('The successfully cropped face output folder is "%s".\n', outputBaseFolder);

% Cleanup - empties the output folders
if ~isempty(outputBaseFolder)
delete([outputBaseFolder filesep '*.*']);
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

%% Read input image and detect face
for i=1:height(fileTable)
       fileName=fileTable.fullFileName{i};
       fprintf('Processing image (%d of %d): %s...', i,height(fileTable),fileName);
       
       % Read Image
       I = imread(fileName);
       
       % Perform Detection
        kernel = [ -1 -1 -1; -1 8 -1; -1 -1 -1];
        filteredImage = imfilter(I, kernel, 'same');
       % Output image of detected Face Area
       outputFileName=[outputBaseFolder filesep 'image_' num2str(i) '.pgm'];
       imwrite(filteredImage, outputFileName, 'pgm')

       fprintf('Done. \n');
end
