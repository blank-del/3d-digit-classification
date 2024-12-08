function classifier_final()
    % classifier_final
    %   Trains the neural network on the processed data, and saves the
    %   model
    %
    %   Example:
    %       result = classifier();
    %
    %   This function doesn't return anything but saves the model output in
    %   two files; one for model output with file name as
    %   "net_digit_classify.mat" and the layers as "layers.mat"
    clearvars; 
    close all; 
    clc; 
    clear all;
    % parameter to 
    rng(42);
    
    % process raw data and call the process data function if the .mat file for
    % processed data doesn't exist
    processed_data_location = 'data_processed_v2.mat';
    if exist(processed_data_location, 'file') == 0
            disp('Processed file not found, cooking data now!');
            processor_m1_v2();
    end
    load(processed_data_location);
    
    X_data = X;
    Y_data = Y;
    
    % assuming the data is in the shape (3, 300, 1000)
    num_samples = size(X_data, 3); % data samples
    
    
    % defining the layers for the neural network
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
    
    
    X_train = X_data;
    Y_train = Y_data;
    
    % reshaping the data to make it compatible for the neural network
    X_train = permute(X_train, [2, 1, 3]); 
    X_train = reshape(X_train, [300, 3, 1, num_samples]);  
    
    
    % DEFINE TRAINING PARAMETERS
    iteration = 0;
    MaxEpochs = 10;
    miniBatchSize = 32;
    LR = 0.001;
    EpochCounter = 0;
    classes = categories(Y_train);
    numClasses = numel(classes);
    
    numObservations = numel(Y_train);
    numIterationsPerEpoch = floor(numObservations./miniBatchSize);
    numIterations = MaxEpochs * numIterationsPerEpoch;
    
    % INITIALIZE NETWORK WITH LAYERS
    net = dlnetwork(layers);
    
    % INITIALIZE MONITOR TO FOLLOW TRAINING
    monitor = trainingProgressMonitor(Metrics="TrainingLoss",Info="Epoch",XLabel="Iteration");
    
    % INITIALIZE LOSS-FUNCTION GRADIENTS
    averageGrad = [];
    averageSqGrad = [];
   
    % BEGIN THE MAIN TRAINING LOOP
    while EpochCounter < MaxEpochs && ~monitor.Stop
        EpochCounter = EpochCounter + 1;
        
        % Shuffle the data before every epoch to improve generalisation
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
    
            % Update the network parameters using the Adaptive Moment Estimation (Adam) -opimization.
            [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iteration);
            
    
            % Update the training progress monitor.
            recordMetrics(monitor,iteration,TrainingLoss=loss);
            updateInfo(monitor,Epoch=EpochCounter + " of " + MaxEpochs);
            monitor.Progress = 100 * iteration/numIterations;
        end
    end
    
    
    % Save the model and layers architechture
    save("model/net_digit_classify.mat", "net");
    save("model/layers.mat", "layers");

end

% Define the loss function (CROSS-ENTROPY LOSS)
function [loss,gradients,state] = modelLoss(net,X,T)

    % Get predictions by forwarding data through network.
    [Y,state] = forward(net,X);
    
    % Calculate cross-entropy loss.
    loss = crossentropy(Y,T);
    
    % Calculate gradients of loss with respect to learnable parameters.
    gradients = dlgradient(loss,net.Learnables);
end