clear all; close all; clc;

% params
numPoints = 300; 
dataDir = 'data'; 
classes = 0:9; 
numClasses = length(classes);


% script
X = []; 
Y = []; 

for classIdx = 1:numClasses
    label = classes(classIdx);
    folderPath = fullfile(dataDir, num2str(label)); 
    files = dir(fullfile(folderPath, '*.csv'));
    
    for file = files'
        filePath = fullfile(file.folder, file.name);
        points = readmatrix(filePath);
        
        points = points - mean(points, 1);
        scale = max(sqrt(sum(points.^2, 2))); 
        points = points / scale;  

        if size(points, 1) > numPoints
            points = points(1:numPoints, :); 
        elseif size(points, 1) < numPoints
            padding = zeros(numPoints - size(points, 1), 3);  
            points = [points; padding]; 
        end
        
        X = cat(3, X, points'); 
        Y = [Y; label];
    end
end

Y = categorical(Y);
data_mean = mean(X, [1, 2]);
data_std = std(X, 0, [1, 2]);
X = (X - data_mean) ./ data_std; 


% results
save('data_processed.mat', 'X', 'Y', 'data_mean', 'data_std');
