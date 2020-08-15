fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset') ;
imds = imageDatastore('F:\Program Files\MATLAB\R2019a\toolbox\nnet\nndemos\nndatasets\DigitDataset','IncludeSubFolders',true,'labelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.20,'randomized');

myNet = [ imageInputLayer([28 28 1])
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(100) 
    reluLayer 
    fullyConnectedLayer(10) 
    softmaxLayer
     classificationLayer ];
options = trainingOptions('sgdm','InitialLearnRate',0.001,'MiniBatchSize',200,'ValidationPatience',Inf,'MaxEpochs',30,'Plots','training-progress');
trainedNet=trainNetwork(imdsTrain,myNet,options);
%% Classify 
YPred = classify(trainedNet,imdsValidation);
YTest = imdsValidation.Labels;
accuracy = sum(YPred == YTest)/numel(YTest)
