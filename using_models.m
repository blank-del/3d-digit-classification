clc
clearvars

C_pred = zeros(1000,1);
C_data = zeros(1000,1);

% LOAD YOUR TESTING DATA
% the provided testing data is also the training data!!!
Data = load("testdata/rawPointClouds.mat"); 
Data = Data.raw_pointclouds;

% DEFINE YOUR LABEL DATA (TRUE CLASSES)
% the provided testing labels are also the training labels!!!
cc = ones(100,1);
C_data = [zeros(100,1);cc;2*cc;3*cc;4*cc;5*cc;6*cc;7*cc;8*cc;9*cc];


% Iterate through the provided data and save the classifier results
for i = 1:numel(Data)
    pos = Data{i};
    label = digit_classify(pos);
    C_pred(i) = label;
end

accuracy = sum(C_pred == double(C_data)) / numel(C_data);
confmat = confusionmat(C_data, C_pred);

figure;
confusionchart(confmat);
title('confusion matrix');

%% FOR ONE FILE

% points = csvread('/path/to/your/test/file');
% prediction = digit_classifC_v2(points);
% disp(['Prediction is: ', num2str(prediction)]);