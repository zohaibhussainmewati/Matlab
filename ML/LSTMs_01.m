% Load stock market data (replace 'stockData.csv' with your data file)
data = readtable("D:\Project & Research\Dataset\symbols_valid_meta.csv", 'VariableNamingRule', 'preserve');

% Automatically get variable names
varNames = data.Properties.VariableNames;

% Use the variable names to select data
X = data(:, varNames(1:end-1)); % Use all variables except the last one as features

% Replace 'VarN' with the actual variable name in your data that you want to predict
Y = data.(varNames{end}); % Use the last variable as the response

% Define LSTM architecture
numFeatures = size(X, 2);
numResponses = size(Y, 2);
numHiddenUnits = 100;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

% Train LSTM network
net = trainNetwork(X,Y,layers,options);

% Use the trained model to make predictions
predictions = predict(net, X);

% Analyze predictions
% This could involve various steps depending on your goal
% For example, you might want to calculate the accuracy of the model
accuracy = sum(predictions == Y) / length(Y);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
