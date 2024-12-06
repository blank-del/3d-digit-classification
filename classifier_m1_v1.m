clear all; close all; clc;

% Parameters
rng(42);

% Load preprocessed data
load('data_processed.mat'); 

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

% Cross-validation loop
for fold = 1:k
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
    net = trainNetwork(X_train, Y_train, layers, options);
    
    % Evaluate the model
    Y_pred = classify(net, X_test);
    accuracies(fold) = sum(Y_pred == Y_test) / numel(Y_test);
    
    confmat = confusionmat(Y_test, Y_pred);
    confmats{fold} = confmat;
    
    for c = 1:10
        TP = confmat(c, c);
        FP = sum(confmat(:, c)) - TP;
        FN = sum(confmat(c, :)) - TP;
        TN = sum(confmat(:)) - (TP + FP + FN);
        
        specificity(fold, c) = TN / (TN + FP);
        sensitivity(fold, c) = TP / (TP + FN);
        precision(fold, c) = TP / (TP + FP);
        f1(fold, c) = 2 * (precision(fold, c) * sensitivity(fold, c)) / (precision(fold, c) + sensitivity(fold, c));
    end
end

% Average results
accuracy_avg = mean(accuracies);
confmat_avg = sum(cat(3, confmats{:}), 3); 
specificity_avg = mean(specificity, 1);
sensitivity_avg = mean(sensitivity, 1);
precision_avg = mean(precision, 1);
f1_avg = mean(f1, 1);

fprintf('Accuracy: %.2f%%\n', accuracy_avg * 100);
fprintf('Specificity:\n'); disp(specificity_avg);
fprintf('Sensitivity:\n'); disp(sensitivity_avg);
fprintf('Precision:\n'); disp(precision_avg);
fprintf('F1 score:\n'); disp(f1_avg);

% Save the trained model
save('trained_model.mat', 'net');
