close all;
clear all;

%%loading data
%importdata = importdata('./ReferencyjneDane/101/AfterClustering.txt');
importdata = importdata('./ReferencyjneDane2/101/ConvertedQRSRawData_2.txt');
load('./ReferencyjneDane2/101/Class_IDs_2.txt');
data=importdata(:,2:18);

Class_IDs = Class_IDs_2;

%%Normalization
for j = 1:size(data,2)
    vec = data(:,j);
    vec = vec - mean(vec);
    vec = vec/std(vec);
    data(:,j) = vec;
end

train_data_percentage = 0.7;
train_data_amount = floor(train_data_percentage * size(data,1));
train_data = data(1:train_data_amount,:);
test_data = data((train_data_amount+1):size(data,1),:);

%
% tic()
% Mdl = fitcknn(train_data, Class_IDs(1:train_data_amount), 'NumNeighbors', 5, 'Distance', 'minkowski', 'Standardize', 0);
% correct = 0;
% wrong = 0;
% classes1 = zeros(size(test_data, 1), 1);
% for j=1:size(test_data,1)
%     v = Mdl.predict(data(train_data_amount + j, :));
%     classes1(j) = v;
%     if Class_IDs(train_data_amount + j) == v
%         correct = correct + 1;
%     else
%         wrong = wrong + 1;
%     end
% end
% correct_detections_percent1 = correct/(correct+wrong)
% toc()
%%
% tic()
% classes2 = knn(train_data, Class_IDs(1:train_data_amount), test_data, 5);
% correct = 0;
% wrong = 0;
% for j=1:size(test_data,1)
%     if Class_IDs(train_data_amount + j) ==  classes2(j)
%         correct = correct + 1;
%     else
%         wrong = wrong + 1;
%     end
% end 
% correct_detections_percent2 = correct/(correct+wrong)
% classes_missmatch = sum(classes1 ~= classes2) / length(classes1)
% toc()
%% enn

% train_amount = size(train_data, 1);
% classes = Class_IDs(1:train_data_amount);
num_neighbours = 5;

%% compute T_i

%dist_fcn = @(train, test) sqrt(sum(abs(ones(train_amount, 1) * test - train) .^ 2, 2)));
 
%uniq_classes = unique(classes);
%uniq_classes = sort(uniq_classes,1,'ascend');
%uniq_classes_num = length(uniq_classes);





%predict classes
classes3 = zeros(size(test_data, 1), 1);
for e = 1:size(test_data, 1)
    tic()
    Class_IDs(train_data_amount + e)
    c = enn(train_data, Class_IDs(1:train_data_amount), test_data(e,:),num_neighbours);
    classes3(e) = c;
    toc()
end
correct = 0;
wrong = 0;
for j=1:size(test_data,1)
    if Class_IDs(train_data_amount + j) ==  classes3(j)
        correct = correct + 1;
    else
        wrong = wrong + 1;
    end
end
correct_detections_percent3 = correct/(correct+wrong)
classes_missmatch = sum(classes1 ~= classes3) / length(classes1)

%%
clc
addpath(genpath('haibo_he'));
[classes3] = ENN(train_data, Class_IDs(1:train_data_amount), test_data, 5);
expected_labels = Class_IDs(train_data_amount+1:train_data_amount + length(test_data));
correct = sum(expected_labels == classes3);
wrong = length(expected_labels) - correct;
correct_detections_percent3 = correct/(correct+wrong)
classes_missmatch = sum(classes1 ~= classes3) / length(classes1)

uniq_input = unique(Class_IDs(train_data_amount+1:end))
uniq_knn = unique(classes1)
uniq_enn = unique(classes3)
