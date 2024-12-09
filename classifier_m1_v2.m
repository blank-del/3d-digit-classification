clearvars; close all; clc; clear all;


% parameters
 rng(42);
%rng(60);

% process raw data
processed_data_location = 'data_processed_v2.mat';
if exist(processed_data_location, 'file') == 0
        disp('Processed file not found, cooking data now!');
        processor_m1_v2();
end
load('data_processed_v2.mat');

X_data = X;
Y_data = Y;


k = 5;

num_samples = size(X_data, 3); % data samples
indices = crossvalind('Kfold', num_samples, k);

accuracies = zeros(k, 1);
confmats = cell(k, 1);
specificity = zeros(k, 10);
sensitivity = zeros(k, 10);
precision = zeros(k, 10);
accuracy = zeros(k, 10);
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
];



% ITERATE THROUGH K-FOLD CROSS-VALIDATION
for fold = 1:k

    fprintf('training fold %d of %d...\n', fold, k);
    Idx_test = (indices == fold);
    Idx_train = ~Idx_test;
    
    X_train = X_data(:, :, Idx_train);
    Y_train = Y_data(Idx_train);

    X_test = X_data(:, :, Idx_test);
    Y_test = Y_data(Idx_test);
    
    X_train = permute(X_train, [2, 1, 3]); 
    X_train = reshape(X_train, [300, 3, 1, sum(Idx_train)]); 

    X_test = permute(X_test, [2, 1, 3]); 
    X_test = reshape(X_test, [300, 3, 1, sum(Idx_test)]);  


    % DEFINE TRAINING PARAMETERS
    iteration = 0;
    MaxEpochs = 5;
    miniBatchSize = 32;
    LR = 0.001;
    ValidationData = {X_test, Y_test};
    ValidationFrequency = 10;
    ValidationCounter = 0;
    EpochCounter = 0;
    classes = categories(Y_train);
    numClasses = numel(classes);

    numObservations = numel(Y_train);
    numIterationsPerEpoch = floor(numObservations./miniBatchSize);
    numIterations = MaxEpochs * numIterationsPerEpoch;


    
    % INITIALIZE NETWORK WITH LAYERS
    net = dlnetwork(layers);
    

    % INITIALIZE MONITOR TO FOLLOW TRAINING
    monitor = trainingProgressMonitor(Metrics=["TrainingLoss", "ValidationLoss"],Info="Epoch",XLabel="Iteration");
    groupSubPlot(monitor,"Loss",["TrainingLoss", "ValidationLoss"]);
    


    % INITIALIZE LOSS-FUNCTION GRADIENTS
    averageGrad = [];
    averageSqGrad = [];



    % Convert validation data to a dlarray.
    Xt = dlarray(single(X_test),"SSCB");
    
    % Format validation data classes
    Tt = zeros(numClasses, length(Y_test), "single"); % USES 200 AS LENGHT OF VALIDATION DATA
    for c = 1:numClasses
        Tt(c,Y_test==classes(c)) = 1;
    end

    % If training on a GPU, then convert validation data to a gpuArray.
    if canUseGPU
        Xt = gpuArray(Xt);
    end

    
    tic;

    % BEGIN THE MAIN TRAINING LOOP
    while EpochCounter < MaxEpochs && ~monitor.Stop
        EpochCounter = EpochCounter + 1;
    
        % Shuffle data.
        idx = randperm(numel(Y_train));
        X_train = X_train(:,:,:,idx);
        Y_train = Y_train(idx);

    
        i = 0;
        while i < numIterationsPerEpoch && ~monitor.Stop
            
            i = i + 1;
            iteration = iteration + 1;
    
            % Read mini-batch of data and convert the labels to dummy
            % variables.
            idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
            X = X_train(:,:,:,idx);
    
            % Get the classes corresponding to the minibatch samples
            T = zeros(numClasses, miniBatchSize, "single");
            for c = 1:numClasses
                T(c,Y_train(idx)==classes(c)) = 1;
            end
    
            % Convert mini-batch of data to a dlarray.
            X = dlarray(single(X),"SSCB");
    
            % If training on a GPU, then convert data to a gpuArray.
            if canUseGPU
                X = gpuArray(X);
            end
    
    
            % Evaluate the model loss and gradients using dlfeval and the
            % modelLoss function.
            [loss,gradients,state] = dlfeval(@modelLoss,net,X,T);
            net.State=state; % THIS UPDATES THE BATCHNORM LAYERS!!!


            % Evaluate the model loss for validation data using dlfeval and the
            % modelLoss function.
            ValidationCounter = ValidationCounter + 1;
            if ValidationCounter >= ValidationFrequency

                [validationLoss, grad] = dlfeval(@modelLoss,net,Xt,Tt);
                ValidationCounter = 0;

                % update the validation loss to the trainign monitor
                recordMetrics(monitor,iteration,ValidationLoss=validationLoss);

            end
    
            % Update the network parameters using the Adaptive Moment Estimation (Adam) -opimization.
            [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iteration);
            

            % Update the training progress monitor.
            recordMetrics(monitor,iteration,TrainingLoss=loss);
            updateInfo(monitor,Epoch=EpochCounter + " of " + MaxEpochs);
            monitor.Progress = 100 * iteration/numIterations;
        end
    end
    
    time_training = toc;

Y_pred = predict(net, X_test);
[Y_pred, rows] = find(Y_pred' == max(Y_pred'));
% predictions are from 1-10
Y_pred = categorical(Y_pred);

% creating confustion matrix
confmat = confusionmat(Y_test, Y_pred);

% since the matrix is 11 by 11 it has one row of zeros in the start column
% and zeros in the end row
confmat = confmat(1:end-1, 2:end);
confmats{fold} = confmat;

accuracies(fold) = trace(confmat) / sum(confmat(:));

    for le = 1:10
        TP = confmat(le, le);
        FP = sum(confmat(:, le)) - TP;
        FN = sum(confmat(le, :)) - TP;
        TN = sum(confmat(:)) - (TP + FP + FN);

        specificity(fold, le) = TN / (TN + FP);
        % sensitivity/recall
        sensitivity(fold, le) = TP / (TP + FN);
        % precision
        precision(fold, le) = TP / (TP + FP);
        % f1 score
        f1(fold, le) = 2 * (precision(fold, le) * sensitivity(fold, le)) / (precision(fold, le) + sensitivity(fold, le));
        % accuracy
        accuracy(fold, le) = (TP)/(TP + FP + FN);
    end
end

% Save the model and layers architechture
% save("net_newdata.mat", "net")
% save("layers.mat", "layers")


% results
% metrics efficacy


confmat_avg = sum(cat(3, confmats{:}), 3);
accuracy_avg = trace(confmat_avg)/sum(confmat_avg(:));
mean_accuracies = mean(accuracy, 1);
specificity_avg = mean(specificity, 1);
sensitivity_avg = mean(sensitivity, 1);
precision_avg = mean(precision, 1);
f1_avg = mean(f1, 1);
 
fprintf('accuracy: %.2f%%\n', accuracy_avg * 100);
fprintf('specificity:\n'); disp(specificity_avg);
fprintf('sensitivity:\n'); disp(sensitivity_avg);
fprintf('precision:\n'); disp(precision_avg);
fprintf('f1 score:\n'); disp(f1_avg);

disp(accuracies);
plot(1:k, accuracies, '-o');
xlabel('Fold');
ylabel('Accuracy');
title('Fold-wise Accuracy');

figure;
confusionchart(confmat_avg);
title('confusion matrix');

figure;
bar(mean_accuracies);
title('accuracy');
ylabel('accuracy');
xlabel('class');

figure;
bar(specificity_avg);
title('specificity');
ylabel('specificity');
xlabel('class');

figure;
bar(sensitivity_avg);
title('sensitivity');
ylabel('sensitivity');
xlabel('class');

figure;
bar(precision_avg);
title('precision');
ylabel('precision');
xlabel('class');

figure;
bar(f1_avg);
title('f1 score');
ylabel('f1 score');
xlabel('class');

% metrics efficiency
% fprintf('fold %d training time: %.2f seconds\n', fold, time_training);
% fprintf('peak GPU memory usage: %.2f MB\n', info_gpu.MemoryUsed / 1e6);

% example classifications
fprintf('example classifications...');

num_examples = 10;
Idx_random = randperm(length(Y_test), num_examples);

labels_true = zeros(num_examples, 1);
labels_pred = zeros(num_examples, 1);

for i = 1:num_examples
    Idx_sample = Idx_random(i);
    labels_true(i) = Y_test(Idx_sample);
    labels_pred(i) = Y_pred(Idx_sample);
end

table_examples = table(labels_true, labels_pred);

disp(table_examples);

% Define the loss function (CROSS-ENTROPY LOSS)
function [loss,gradients,state] = modelLoss(net,X,T)

    % Get predictions by forwarding data through network.
    [Y,state] = forward(net,X);
    
    % Calculate cross-entropy loss.
    loss = crossentropy(Y,T);
    
    % Calculate gradients of loss with respect to learnable parameters.
    gradients = dlgradient(loss,net.Learnables);
end