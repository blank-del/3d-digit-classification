% points = csvread('testdata/testdata1h.csv');
% prediction = digit_classify_v2(points);
% disp(['Prediction is: ', num2str(prediction)]);

C_pred = zeros(1000,1);
C_data = zeros(1000,1);

% LOAD YOUR TESTING DATA
% the provided testing data is also the training data!!!
Data = load("testdata/rawPointClouds.mat"); 
Data = Data.raw_pointclouds;

for i = 1:1000
    pos = Data{i};
    label = digit_classify_v2(pos);
    C_pred(i) = label;
    C_data(i) = floor((i-1)/100);
end

accuracies(1) = sum(C_pred == double(C_data)) / numel(C_data);
confmat = confusionmat(C_data, C_pred);

figure;
confusionchart(confmat);
title('confusion matrix');

%% FOR ONE FILE

% points = csvread('/path/to/your/test/file');
% prediction = digit_classifC_v2(points);
% disp(['Prediction is: ', num2str(prediction)]);