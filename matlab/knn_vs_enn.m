close all
clear all

K = 3;
expected_classes_num = 10;
train_data_percentage = 0.7;

files = dir('../data/data_reused_labels/*');
files = files(3:end); %remove . and ..
files_num = length(files);
success_rate = zeros(files_num, 2);
by_class_comp = zeros(files_num, 2, expected_classes_num);
class_freq = zeros(files_num, expected_classes_num);
for i = 1:files_num
    file = files(i, 1).name
    qrs_file = sprintf('../data/data_reused_labels/%s/data.txt', file);
    classes_file = sprintf('../data/data_reused_labels/%s/label.txt', file);
    idata = importdata(qrs_file);
    load(classes_file);

    data=idata(:,2:18);
    num_samples = size(data,1);
    
    Class_IDs = label;
       
    for j = 1:size(data,2)
        vec = data(:,j);
        vec = vec - mean(vec);
        vec = vec/std(vec);
        data(:,j) = vec;
    end
    
    for c = 1:expected_classes_num
        class_freq(i, c) = 100 * sum(Class_IDs(1:num_samples) == c) / num_samples;
    end

    train_data_amount = floor(train_data_percentage * num_samples);
    train_data = data(1:train_data_amount,:);
    train_label = Class_IDs(1:train_data_amount);
    test_data = data((train_data_amount+1):num_samples,:);
    test_data_amount = length(test_data);
    test_label = Class_IDs(train_data_amount+1:train_data_amount+test_data_amount);
    
    tic()
    classes_knn = knn(train_data, train_label, test_data, K);
    knn_time = toc()
    tic()
    classes_enn = enn(train_data, train_label, test_data, K); 
    enn_time = toc()
    
    correct = sum(test_label == classes_knn);
    success_rate(i, 1)  = 100 * correct / test_data_amount;
    correct = sum(test_label == classes_enn);
    success_rate(i, 2)  = 100 * correct / test_data_amount;
    
    % positive -> label != 1
    % negative -> label == 1
%%knn
    Tp = sum(classes_knn ~= 1 & test_label ~= 1);
    Tn = sum(classes_knn == 1 & test_label == 1);
    Fp = sum(classes_knn ~= 1 & test_label == 1);
    Fn = sum(classes_knn == 1 & test_label ~= 1);  

    sensitivity(i,1) = Tp/(Tp+Fn);
    specificity(i,1) = Tn/(Tn+Fp);
    
%%enn
    Tp = sum(classes_enn ~= 1 & test_label ~= 1);
    Tn = sum(classes_enn == 1 & test_label == 1);
    Fp = sum(classes_enn ~= 1 & test_label == 1);
    Fn = sum(classes_enn == 1 & test_label ~= 1);  
    
    sensitivity(i,2) = Tp/(Tp+Fn);
    specificity(i,2) = Tn/(Tn+Fp);
       
    
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