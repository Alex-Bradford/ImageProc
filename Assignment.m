%% load the image set
%% Need to delete the ambient image which is about 300kb in Yale11-39
dataset = imageSet('C:\Users\98114236\Downloads\CroppedYale\CroppedYale','recursive')

%% view the images
%%% display a montage of the first subject
%montage(dataset(1).ImageLocation)
%%% display a single image
%imshow(read(dataset(1),1))

%% split the dataset into training and test sets
[training,test] = partition(dataset,[0.8 0.2]);

%% Feature Extraction
% Search feature extraction in matlab and try out some other feature
% extraction techniques

% Extract HOG features for training set
% creates trainingFeatures matrix which is size "trainingSetSize x
% dimensionality of problem"
% creates trainingLabels array, maps each training datapoint to subject

% We hard code 16560 because that is the size of array produced when
% extracting HOG feature from the 192x168 images
trainingFeatures = zeros(size(training,2)*training(1).Count,16560);
featureCount = 1;
% for each subject...
for i = 1:size(training,2)
    % for each picture of that subject...
    for j = 1:training(i).Count
        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),j));
        trainingLabel{featureCount} = training(i).Description;
        featureCount = featureCount + 1;
    end
    personIndex{i} = training(i).Description;
end
% it produces a 1x1961 trainingLabel because there are 15 'bad' photos in
% trainging(11-17)
% also, the last 15 rows of trainingFeatures are empty

% Remove the last 15 rows otherwise training the classifier will throw an
% error...this takes a while though
trainingFeatures = trainingFeatures(1:end-15,:);

%% Classification
% training a classifier follows the pattern:
% model = fit[c/l][modeltype](X, Y)
% where c/l = defines classification or regression
% modeltype, defines the classifier eg. kNN or SVM
% X = the trainingFeatures
% Y = the labels

% create the classifier
model = fitcecoc(trainingFeatures, trainingLabel);

%% Calculate the accuracy

% for each subject...
correct_pred = 0;
false_pred = 0;
for i = 1:size(test,2)
    % for each picture of that subject...
    for j = 1:test(i).Count
        disp(i)
        disp(j)
        queryImage = read(test(i),j);
        queryFeatures = extractHOGFeatures(queryImage);
        personLabel = predict(model,queryFeatures);
        booleanIndex = strcmp(personLabel, personIndex);
        integerIndex = find(booleanIndex);
        if integerIndex == i
            correct_pred = correct_pred + 1;
        else
            false_pred = false_pred + 1;
        end
        disp("worked")
    end
end

accur = correct_pred/(correct_pred+false_pred)
