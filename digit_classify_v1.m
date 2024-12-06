function C = digit_classify_v1(testdata)
    load('data_processed.mat', 'data_mean', 'data_std'); 
    load('trained_model.mat', 'net'); 
    
    testdata = double(testdata);
    
    testdata = testdata - mean(testdata, 1);  
    scale = max(sqrt(sum(testdata.^2, 2))); 
    testdata = testdata / scale;
    
    num_points = 300;
    [num_rows, num_cols] = size(testdata);

    if num_rows > num_points
        testdata = testdata(1:num_points, :); 
    elseif num_rows < num_points
        padding = zeros(num_points - num_rows, num_cols);  
        testdata = [testdata; padding]; 
    end

    testdata = (testdata - data_mean) ./ data_std;
    
    if size(testdata, 3) > 1
        testdata = testdata(:,:,1); 
    end
    
    testdata = reshape(testdata, [300, 3, 1, 1]);  
    
    C = classify(net, testdata);
    C = double(C); 
    
    C = C - 1;  
end
