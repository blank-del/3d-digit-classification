function C = digit_classify_v2(rawPointCloud)
    % digit_classify
    %   returns the label of number represented as 3D point cloud 
    %   input as double matrix of size N x 3
    %
    %   uses pretrained CNN classifier
    %
    %   Example:
    %       label = digit_classify(rawPointCloud);
    %
    %       rawPointCloud = unedited pointcloud from motion tracking sensor
    %       result is a label for the digit that is drawn
    % 
    % With slow laptop classification of 1000 samples takes approx. 5 min


    % Indicate the location of model parameters
    model_location = 'model/net_digit_classify.mat';
    layers_location = 'model/layers.mat';

    % if the model is not found train the model and save parameters
    if exist(model_location, 'file') == 0
        disp('Model not found, generating model.');
        classifier_final();
    end

    % load model and layers architechture
    load(model_location, 'net');
    load(layers_location, 'layers');
    
    % preprocess the raw point cloud
    testdata = process_data(rawPointCloud); 
    
    % classify
    C = predict(net, testdata);
    C = find(C == max(C));
    
    C = C - 1; % map from classes 1-10 to labels 0-9
end
