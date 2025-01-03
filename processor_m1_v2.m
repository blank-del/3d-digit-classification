function processor_m1_v2()
    % This script combines all the training data files into one .mat file which
    % has applied transforation on the data points and has a shape of
    % (Number of columns * Number of rows * Number of files)

    % params
    numPoints = 300; 
    dataDir = 'training_data_mat'; 
    classes = 0:9; 
    numClasses = length(classes);


    % script
    X = []; 
    Y = []; 

    for classIdx = 1:numClasses
        label = classes(classIdx);
        folderPath = fullfile(dataDir, num2str(label)); 
        files = dir(fullfile(folderPath, '*.mat'));
        
        for file = files'
            filePath = fullfile(file.folder, file.name);
            points = load(filePath);
            
            points = points.pos;
            points = process_data(points);

            X = cat(3, X, points'); 
            Y = [Y; label];
        end
    end

    Y = categorical(Y);

    % results
    save('data_processed_v2.mat', 'X', 'Y');
end