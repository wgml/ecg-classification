close all;
clear all;

[time, signals, classes, stamps] = read_fcn('/home/vka/Programming/C/workspace/ecg-classification/data/mit-bih', '100');
% plot_fcn('/home/vka/Programming/C/workspace/ecg-classification/data/mit-bih', '100');

signal = signals(:, 1);

plot(time, signal)

ft = abs(fft(signal)).^2;
figure
loglog(ft)
