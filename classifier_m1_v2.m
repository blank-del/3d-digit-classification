clear all; close all; clc;

% Parameters
rng(42);

% Load preprocessed data
load('data_processed_v2.mat'); 

k = 5;  % Number of folds for cross-validation
num_samples = size(X, 3);
indices = crossvalind('Kfold', num_samples, k); 

accuracies = zeros(k, 1);
confmats = cell(k, 1);
specificity = zeros(k, 10);
sensitivity = zeros(k, 10);
precision = zeros(k, 10); 
f1 = zeros(k, 10);

layers = [
    imageInputLayer([300, 3, 1], 'Name', 'input', 'Normalization', 'none')
    convolution2dLayer([3, 3], 64, 'Stride', [1, 1], 'Name', 'conv1', 'Padding', 'same')
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.2, 'Name', 'dropout1')
    convolution2dLayer([3, 3], 128, 'Stride', [1, 1], 'Name', 'conv2', 'Padding', 'same')
    batchNormalizationLayer('Name', 'batchnorm2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([2, 2], 'Name', 'maxpool', 'Stride', [2, 2])
    fullyConnectedLayer(256, 'Name', 'fc1')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.3, 'Name', 'dropout2')
    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(10, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];



fprintf('Training fold %d of %d...\n', fold, k);

Idx_test = (indices == fold);
Idx_train = ~Idx_test;

X_train = X(:, :, Idx_train);
Y_train = Y(Idx_train);
X_test = X(:, :, Idx_test);
Y_test = Y(Idx_test);

% Reshape and normalize data for training and testing
X_train = permute(X_train, [2, 1, 3]); 
X_train = reshape(X_train, [300, 3, 1, sum(Idx_train)]); 
X_train = normalize(X_train, 'range');

X_test = permute(X_test, [2, 1, 3]); 
X_test = reshape(X_test, [300, 3, 1, sum(Idx_test)]); 
X_test = normalize(X_test, 'range'); 

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-3, ...
    'ValidationData', {X_test, Y_test}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train the network
net = dlnetwork(layers);
net = initialize(net);
%net = trainNetwork(X_train, Y_train, layers, options);

% Evaluate the model
Y_pred = classify(net, X_test);
accuracies(fold) = sum(Y_pred == Y_test) / numel(Y_test);

% Save the trained model
% save('trained_model.mat', 'net');



function [loss,gradients,state] = modelLoss(net,X,T)

% Forward data through network.
[Y,state] = forward(net,X);

% Calculate cross-entropy loss.
loss = crossentropy(Y,T);

% Calculate gradients of loss with respect to learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end

function parameters = sgdStep(parameters,gradients,learnRate)

parameters = parameters - learnRate .* gradients;

end