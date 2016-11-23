close all;
clear all;

%%loading data
%importdata = importdata('./ReferencyjneDane/101/AfterClustering.txt');
importdata = importdata('./ReferencyjneDane/103/ConvertedQRSRawData.txt');
load('./ReferencyjneDane/103/Class_IDs.txt');
data=importdata(:,2:18);
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

%%
tic()
Mdl = fitcknn(train_data, Class_IDs(1:train_data_amount), 'NumNeighbors', 5, 'Distance', 'minkowski', 'Standardize', 0);
correct = 0;
wrong = 0;
classes1 = zeros(size(test_data, 1), 1);
for j=1:size(test_data,1)
    v = Mdl.predict(data(train_data_amount + j, :));
    classes1(j) = v;
    if Class_IDs(train_data_amount + j) == v
        correct = correct + 1;
    else
        wrong = wrong + 1;
    end
end
correct_detections_percent1 = correct/(correct+wrong)
toc()
%%
tic()
classes2 = knn(train_data, Class_IDs(1:train_data_amount), test_data, 5);
correct = 0;
wrong = 0;
for j=1:size(test_data,1)
    if Class_IDs(train_data_amount + j) ==  classes2(j)
        correct = correct + 1;
    else
        wrong = wrong + 1;
    end
end
correct_detections_percent2 = correct/(correct+wrong)
classes_missmatch = sum(classes1 ~= classes2) / length(classes1)
toc()
%% enn
classes3 = zeros(size(test_data, 1), 1);
for e = 1:size(test_data, 1)
    tic()
    c = enn(train_data, Class_IDs(1:train_data_amount), test_data(e,:), 5)
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