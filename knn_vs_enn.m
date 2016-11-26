close all
clear all

addpath(genpath('haibo_he'));
K = 5;
expected_classes_num = 3;
train_data_percentage = 0.7;

files = dir('ReferencyjneDane/*');
files = files(3:end); %remove . and ..
files_num = length(files);
success_rate = zeros(files_num, 2);
by_class_comp = zeros(files_num, 2, expected_classes_num);
for i = 1:files_num
    file = files(i, 1).name
    qrs_file = sprintf('./ReferencyjneDane/%s/ConvertedQRSRawData.txt', file);
    classes_file = sprintf('./ReferencyjneDane/%s/Class_IDs.txt', file);
    idata = importdata(qrs_file);
    load(classes_file);
    data=idata(:,2:18);
    for j = 1:size(data,2)
        vec = data(:,j);
        vec = vec - mean(vec);
        vec = vec/std(vec);
        data(:,j) = vec;
    end

    train_data_amount = floor(train_data_percentage * size(data,1));
    train_data = data(1:train_data_amount,:);
    train_label = Class_IDs(1:train_data_amount);
    test_data = data((train_data_amount+1):size(data,1),:);
    test_data_amount = length(test_data);
    test_label = Class_IDs(train_data_amount+1:train_data_amount+test_data_amount);
    
    [train_data_trunc, train_label_trunc] = truncate_train_data(train_data, train_label, expected_classes_num, 3);
    tic()
    classes_knn = knn(train_data, train_label, test_data, K);
    knn_time = toc()
    
    tic()
    classes_enn = ENN(train_data_trunc, train_label_trunc, test_data, K);
    enn_time = toc()
    
    % compare
    correct = sum(test_label == classes_knn);
    success_rate(i, 1)  = correct / test_data_amount;
    correct = sum(test_label == classes_enn);
    success_rate(i, 2)  = correct / test_data_amount;

    class_occurences = zeros(1, expected_classes_num);
    for j = 1:test_data_amount
        class_id = test_label(j);
        class_occurences(class_id) = class_occurences(class_id) + 1;
        if classes_knn(j) == class_id
            by_class_comp(i, 1, class_id) = by_class_comp(i, 1, class_id) + 1;
        end
        if classes_enn(j) == class_id
            by_class_comp(i, 2, class_id) = by_class_comp(i, 2, class_id) + 1;
        end
    end
    for j = 1:expected_classes_num
        if class_occurences(j) == 0
            by_class_comp(i, 1, j) = -1;
            by_class_comp(i, 2, j) = -1;
        else
            by_class_comp(i, 1, j) = 100 * by_class_comp(i, 1, j) / class_occurences(j);
            by_class_comp(i, 2, j) = 100 * by_class_comp(i, 2, j) / class_occurences(j);
        end
    end
end