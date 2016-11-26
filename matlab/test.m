close all;
clear all;

[time, signals, classes, stamps] = read_fcn('/home/vka/Programming/C/workspace/ecg-classification/data/mit-bih', '109');
unique(classes)
% plot_fcn('/home/vka/Programming/C/workspace/ecg-classification/data/mit-bih', '102');
% 
% signal = signals(:, 1);
% plot(signal)