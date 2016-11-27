function [ neighbour_classes_per_node,T,n_i ] = enn_prepare( train_data, train_classes, num_neighbours )
if nargin ~= 4
    p = 2;
    dist_fcn = @(x, y) sum(abs(x - y) .^ p) ^ (1 / p);
end
train_amount = size(train_data, 1);
uniq_classes = unique(train_classes);
uniq_classes = sort(uniq_classes,1,'ascend');
uniq_classes_num = length(uniq_classes);

T = zeros(uniq_classes_num, 1);
neighbour_classes_per_node = zeros(train_amount - 1, num_neighbours);
n_i = zeros(uniq_classes_num, 1);
for class_idx = 1:uniq_classes_num
   class = uniq_classes(class_idx);
   class_idxs = find(train_classes == class);
   n_i(class_idx) = numel(class_idxs);
   for i = 1:numel(class_idxs)
       idx = class_idxs(i);
       x = train_data(idx, :);
       train_data_x = zeros(train_amount - 1, length(x));
       classes_x = zeros(train_amount - 1, 1);
       if (idx > 1)
           train_data_x(1:idx-1, :) = train_data(1:idx-1, :);
           classes_x(1:idx-1) = train_classes(1:idx-1);
       end
       if (idx < train_amount)
           train_data_x(idx:train_amount - 1, :) = train_data(idx+1:end, :);
           classes_x(idx:train_amount - 1) = train_classes(idx+1:end);
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

end

