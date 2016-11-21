close all;
clear all;

%%loading data
%importdata = importdata('./ReferencyjneDane/101/AfterClustering.txt');
importdata = importdata('./ReferencyjneDane/101/ConvertedQRSRawData.txt');
load('./ReferencyjneDane/101/Class_IDs.txt');
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

train_amount = size(train_data, 1);
classes = Class_IDs(1:train_data_amount);
num_neighbours = 5;

% compute T_i

p = 2;
dist_fcn = @(x, y) sum(abs(x - y) .^ p) ^ (1 / p);

uniq_classes = unique(classes);
uniq_classes = sort(uniq_classes,1,'ascend');
uniq_classes_num = length(uniq_classes);

T = zeros(uniq_classes_num, 1);
neighbour_classes_per_node = zeros(train_amount - 1, num_neighbours);
n_i = zeros(uniq_classes_num, 1);
for class_idx = 1:uniq_classes_num
   class = uniq_classes(class_idx);
   class_idxs = find(classes == class);
   n_i(class_idx) = numel(class_idxs);
   for i = 1:numel(class_idxs)
       idx = class_idxs(i);
       x = train_data(idx, :);
       train_data_x = zeros(train_amount - 1, length(x));
       classes_x = zeros(train_amount - 1, 1);
       if (idx > 1)
           train_data_x(1:idx-1, :) = train_data(1:idx-1, :);
           classes_x(1:idx-1) = classes(1:idx-1);
       end
       if (idx < train_amount)
           train_data_x(idx:train_amount - 1, :) = train_data(idx+1:end, :);
           classes_x(idx:train_amount - 1) = classes(idx+1:end);
       end
       x_nn_classes = n_closest_neighbours(train_data_x, classes_x, x, num_neighbours, dist_fcn);
       neighbour_classes_per_node(idx, :) = x_nn_classes;
       for r = 1:num_neighbours
          if (class == x_nn_classes(r))
              T(class_idx) = T(class_idx) + 1;
          end
       end
   end
   T(class_idx) = T(class_idx) / (n_i(class_idx) * num_neighbours);
end

%predict classes
classes3 = zeros(size(test_data, 1), 1);
for e = 1:size(test_data, 1)
    tic()
    Class_IDs(train_data_amount + e)
    c = enn(train_data, Class_IDs(1:train_data_amount), test_data(e,:),num_neighbours,neighbour_classes_per_node,T,n_i)
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
