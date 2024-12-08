points = csvread('testdata/testdata1h.csv');
prediction = digit_classify_v2(points);
disp(['Prediction is: ', num2str(prediction)]);