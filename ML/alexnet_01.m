
% Load the pre-trained AlexNet model
net = alexnet;

% Specify the paths to your training, testing, and prediction datasets
trainDatasetPath = "D:\Project & Research\Dataset\seg\seg_train";
testDatasetPath = "D:\Project & Research\Dataset\seg\seg_test";
predictDatasetPath = "D:\Project & Research\Dataset\seg\seg_pred";

% Create ImageDatastore objects for training, testing, and prediction datasets
imdsTrain = imageDatastore(trainDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest = imageDatastore(testDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsPredict = imageDatastore(predictDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Resize the images to match the input size of the network
imdsTrain.ReadFcn = @(loc)imresize(imread(loc), net.Layers(1).InputSize(1:2));
imdsTest.ReadFcn = @(loc)imresize(imread(loc), net.Layers(1).InputSize(1:2));
imdsPredict.ReadFcn = @(loc)imresize(imread(loc), net.Layers(1).InputSize(1:2));

% Replace the last few layers of the pre-trained model
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

% Specify the training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 6, ...
    'InitialLearnRate', 0.0001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsTest, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
netTransfer = trainNetwork(imdsTrain, layers, options);

% Use the trained network to predict the labels of the test data
YPredTest = classify(netTransfer, imdsTest);

% Calculate the accuracy of the predictions
accuracy = sum(YPredTest == imdsTest.Labels) / numel(imdsTest.Labels);
fprintf('Accuracy of the network on the test images: %.2f %%\n', accuracy * 100);

% Use the trained network to predict the labels of the prediction data
YPredPredict = classify(netTransfer, imdsPredict);
