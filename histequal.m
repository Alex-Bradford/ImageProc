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
uiwait(msgbox('Please select the face output directory.'));
outputBaseFolder = uigetdir(start_path);
if outputBaseFolder == 0
	return;
end
fprintf('The face output folder is "%s".\n', outputBaseFolder);

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




%% Read image and equlaize histogram
for i=1:height(fileTable)
       fileName=fileTable.fullFileName{i};
       fprintf('Processing image (%d of %d): %s...', i,height(fileTable),fileName);
       
       % Read Image
      I = imread(fileName);
       %histogram equalization
      J = histeq(I);

      %write to disk
       outputFileName=[outputBaseFolder filesep 'image_' num2str(i) '.pgm'];
       imwrite(J, outputFileName, 'pgm')

       fprintf('Done. \n');

end