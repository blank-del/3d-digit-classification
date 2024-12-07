clear all; close all; clc;

% params
filename = 'testdata0l.csv'; 
folderPath = 'testdata';   


% script
filePath = fullfile(folderPath, filename);
testdata = readmatrix(filePath);

C = digit_classify_v1(testdata); 


% results
C
