clc
clear all
close all
fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset') ;
imds = imageDatastore('F:\Program Files\MATLAB\R2019a\toolbox\nnet\nndemos\nndatasets\DigitDataset','IncludeSubFolders',true,'labelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.75,'randomized');

myNet = [imageInputLayer([28 28 1])
    convolution2dLayer([3 3],32,'padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 3],64,'padding','same') 
    batchNormalizationLayer
    reluLayer 
    fullyConnectedLayer(10) 
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm','InitialLearnRate',0.1,'MiniBatchSize',200,'ValidationPatience',Inf,'MaxEpochs',20,'Plots','training-progress');
trainedNet=trainNetwork(imdsTrain,myNet,options);
%% Classify 
YPred = classify(trainedNet,imdsValidation);
YTest = imdsValidation.Labels;
accuracy = sum(YPred == YTest)/numel(YTest)
