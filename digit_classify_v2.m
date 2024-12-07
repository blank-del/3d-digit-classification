function C = digit_classify_v1(testdata)
    load('model/trained_model.mat', 'net'); 
    
    
    testdata = reshape(testdata, [300, 3, 1, 1]);  
    
    C = classify(net, testdata);
    C = double(C); 
    
    C = C - 1;  
end
