function C = digit_classify_v1(testdata)
    model_location = 'model/trained_model.mat';

    if exist(model_location, 'file') == 0
        disp('Model not found, generating model.');
        % call to a classification function
    end
    load('model/trained_model.mat', 'net'); 
    
    testdata = process_data(testdata);
    testdata = reshape(testdata, [300, 3, 1, 1]);  
    
    C = classify(net, testdata);
    C = double(C); 
    
    C = C - 1;  
end
