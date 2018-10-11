function f = showTwoImages(imageOne, imageTwo)
    
    subplot(1, 2, 1); 
    imshow(imageOne);
    title('Query Face');
    
    subplot(1, 2, 2);
    imshow(imageTwo);
    title('Prediction');

end
