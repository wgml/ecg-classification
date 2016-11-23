function [ class ] = enn( train_data, classes, test_probe, num_neighbours, dist_fcn )

train_amount = size(train_data, 1);

if nargin ~= 5
    p = 2;
    dist_fcn = @(x, y) sum(abs(x - y) .^ p) ^ (1 / p);
end

uniq_classes = unique(classes);
uniq_classes_num = length(uniq_classes);

% compute T_i
T = zeros(uniq_classes_num, 1);
neighbour_classes_per_node = zeros(train_amount, num_neighbours);
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
           classes_x(1:idx-1) = classes(1:idx-1, :);
       end
       if (idx < train_amount)
           train_data_x(idx:train_amount - 1, :) = train_data(idx+1:end, :);
           classes_x(idx:train_amount - 1) = classes(idx+1:end, :);
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

% compute delta_n_i^j
% delta_n(i,j) = delta_n_i^j
delta_n = zeros(uniq_classes_num);
for j = 1:uniq_classes_num
   class = uniq_classes(class_idx);
   train_data_j = [train_data; test_probe];
   class_idxs = find(classes == class);
   classes_j = [classes; class];
   n_i(class_idx) = numel(class_idxs);
   neighbour_classes_per_node_j = zeros(train_amount, num_neighbours);

   for i = 1:numel(class_idxs)
       idx = class_idxs(i);
       x = train_data_j(idx, :);
       train_data_x = zeros(train_amount, length(x));
       classes_x = zeros(train_amount - 1, 1);
       if (idx > 1)
           train_data_x(1:idx-1, :) = train_data_j(1:idx-1, :);
           classes_x(1:idx-1) = classes_j(1:idx-1, :);
       end
       if (idx < train_amount)
           train_data_x(idx:train_amount, :) = train_data_j(idx+1:end, :);
           classes_x(idx:train_amount) = classes_j(idx+1:end, :);
       end
       x_nn_classes = n_closest_neighbours(train_data_x, classes_x, x, num_neighbours, dist_fcn);
       neighbour_classes_per_node_j(idx, :) = x_nn_classes;
   end
   
   for i = 1:uniq_classes_num
       for e = 1:numel(class_idxs)
           n_org = sum(neighbour_classes_per_node(e, :) == class);
           n_new = sum(neighbour_classes_per_node_j(e, :) == class);
           
           if i == j && n_new > n_org
               delta_n(i, j) = delta_n(i, j) + 1;
           elseif i ~= j && n_new < n_org
               delta_n(i, j) = delta_n(i, j) + 1;
           end
       end
   end
end

% determine result class
probe_neighbours = n_closest_neighbours(train_data, classes, test_probe, num_neighbours, dist_fcn);
k_i = zeros(uniq_classes_num, 1);
for c_idx = 1:uniq_classes_num
   k_i(c_idx) = sum(probe_neighbours == uniq_classes(c_idx));
end
predictions = zeros(uniq_classes_num, 1);
for j = 1:uniq_classes_num
    s1 = (delta_n(j, j) + k_i(j) - num_neighbours * T(j)) / ((n_i(j) + 1) / num_neighbours);
    s2 = 0;
    for i = 1:uniq_classes_num
        if i == j
            continue
        end
        s2 = s2 + delta_n(i,j) / (n_i(i) * num_neighbours);
    end
    predictions(j) = s1 - s2;
end
[~, c_idx] = max(predictions);
class = uniq_classes(c_idx);