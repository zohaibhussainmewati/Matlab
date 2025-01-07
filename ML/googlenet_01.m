% Load pre-trained GoogLeNet network
net = googlenet;

% Load training images
imdsTrain = imageDatastore("D:\Project & Research\Dataset\seg\seg_train", 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Load test images
imdsTest = imageDatastore("D:\Project & Research\Dataset\seg\seg_test", 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Resize all training images to [224 224] for GoogLeNet network architecture
augimdsTrain = augmentedImageDatastore([224 224], imdsTrain);
augimdsTest = augmentedImageDatastore([224 224], imdsTest);

% Extract the layer graph from the trained network 
lgraph = layerGraph(net); 

% Find the names of the two layers to replace
learnableLayer = lgraph.Layers(end-2);
classLayer = lgraph.Layers(end);

% Replace the fully connected layer with a new one.
numClasses = numel(categories(imdsTrain.Labels));
newLearnableLayer = fullyConnectedLayer(numClasses, 'Name','new_fc', 'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnableLayer);

% Replace the classification layer
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph, classLayer.Name, newClassLayer);

% Train network
options = trainingOptions('sgdm', 'MiniBatchSize',10, 'MaxEpochs',6, 'InitialLearnRate',1e-4, 'Shuffle','every-epoch', 'ValidationData',augimdsTest, 'ValidationFrequency',3, 'Verbose',false, 'Plots','training-progress');
net = trainNetwork(augimdsTrain,lgraph,options);

% Classify the test images using the fine-tuned network
[YPred,scores] = classify(net,augimdsTest);

% Calculate the classification accuracy on the test set. 
accuracy = mean(YPred == imdsTest.Labels);