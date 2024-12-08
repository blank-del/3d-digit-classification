clc
close all
clearvars

load("net_newdata.mat")
load("layers.mat")

load('data_processed_v2.mat')

model = 1


if exist("data")
    X_data = data;
    Y_data = labels;
    Y_data = categorical(Y_data);
elseif exist("X") && exist("Y")
    X_data = X;
    Y_data = Y;
end

X_data = permute(X_data, [2, 1, 3]); 
X_data = reshape(X_data, [300, 3, 1, 1000]); 
X_data = normalize(X_data, 'range');


X_data = dlarray(single(X_data),"SSCB");



Y_pred = predict(net, X_data);


Y_pred = single(extractdata(Y_pred))';

[Y_pred, rows] = find(Y_pred' == max(Y_pred'));
accuracies(model) = sum(Y_pred == double(Y_data)) / numel(Y_data);
Y_pred = categorical(Y_pred);
confmat = confusionmat(Y_data, Y_pred);
%confmat = confmat(1:end-1, 2:end);
confmats{model} = confmat;


figure;
confusionchart(confmats{1});
title('confusion matrix');
