function [ result ] = enn( train_data, classes, test_data, num_neighbours, dist_fcn )


train_amount = size(train_data, 1);
test_amount = size(test_data, 1);
result = zeros(test_amount, 1);

if nargin ~= 5
    p = 2;
    dist_fcn = @(x, y) sum(abs(x - y) .^ p) ^ (1 / p);
end

uniq_classes = unique(classes);
uniq_classes_num = length(uniq_classes);

T = zeros(uniq_classes_num, 1);
for class_idx = 1:uniq_classes_num
   class = uniq_classes(class_idx);
   class_idxs = find(classes == class);
   n_i = numel(class_idxs);
   for i = 1:numel(class_idxs)
       idx = class_idxs(i);
       x = train_data(idx, :);
       train_data_x = [];
       classes_x = [];
       if (idx > 1)
           train_data_x = [train_data(1:idx-1, :)];
           classes_x = [classes(1:idx-1, :)];
       end
       if (idx < train_amount)
           train_data_x = [train_data_x; train_data(idx+1:end, :)];
           classes_x = [classes_x; classes(idx+1:end, :)];
       end
       for r = 1:num_neighbours
          x_nn_class = knn(train_data_x, classes_x, x, 1, dist_fcn);
       
          if (class == x_nn_class)
              T(class_idx) = T(class_idx) + 1;
          end
       end
   end
   T(class_idx) = T(class_idx) / (n_i * num_neighbours);
end
T