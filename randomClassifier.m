imgSet = imageSet('C:\Users\ramuu\projects\Image Recognition\ImageProc\Yale FaceDatabase\CroppedYale', 'recursive');
labelCount = countEachLabel(imgSet);
numberOfLabels = length(imgSet);
correct_pred = 0;
false_pred = 0;
for i = 1:numberOfLabels
    for j = 1:imgSet(i).Count
            randomLabelIndex = randi([1 numberOfLabels],1);
            if randomLabelIndex == i
                correct_pred = correct_pred + 1;
            else
                false_pred = false_pred + 1;
            end
    end
end
accur = correct_pred/(correct_pred+false_pred)

        

