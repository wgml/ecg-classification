function [ test_label ] = enn( train_data, classes, test_data, num_neighbours,neighbour_classes_per_node,T,n_i, dist_fcn )

train_amount = size(train_data, 1);

if nargin ~= 8
    p = 2;
    dist_fcn = @(x, y) sum(abs(x - y) .^ p) ^ (1 / p);
end
 
uniq_classes = unique(classes);
uniq_classes = sort(uniq_classes,1,'ascend');
uniq_classes_num = length(uniq_classes);

% compute delta_n_i^j
% delta_n(i,j) = delta_n_i^j
delta_n = zeros(uniq_classes_num);
for probe = 1:size(test_data, 1)
    test_probe = test_data(probe, :);
    for j = 1:uniq_classes_num
       class_idx = j;
       test_label = uniq_classes(class_idx);
       train_data_j = [train_data; test_probe];
       class_idxs = find(classes == test_label);
       classes_j = [classes; test_label];
       neighbour_classes_per_node_j = zeros(train_amount, num_neighbours);

       for i = 1:numel(class_idxs)
           idx = class_idxs(i);
           x = train_data_j(idx, :);
           train_data_x = zeros(train_amount, length(x));
           classes_x = zeros(train_amount , 1);
           if (idx > 1)
               train_data_x(1:idx-1, :) = train_data_j(1:idx-1, :);
               classes_x(1:idx-1) = classes_j(1:idx-1);
           end
           if (idx < train_amount)
               train_data_x(idx:train_amount, :) = train_data_j(idx+1:end, :);
               classes_x(idx:train_amount) = classes_j(idx+1:end);
           end
           x_nn_classes = n_closest_neighbours(train_data_x, classes_x, x, num_neighbours, dist_fcn);
           neighbour_classes_per_node_j(idx, :) = x_nn_classes;
       end

       for i = 1:uniq_classes_num
           for e = 1:numel(class_idxs)
               n_org = sum(neighbour_classes_per_node(e, :) == uniq_classes(i));
               n_new = sum(neighbour_classes_per_node_j(e, :) == uniq_classes(i));

               if i == j && n_new > n_org
                   delta_n(i, j) = delta_n(i, j) + 1;
               elseif i ~= j && n_new < n_org
                   delta_n(j, i) = delta_n(j, i) + 1;
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
        s1 = (delta_n(j, j) + k_i(j) - num_neighbours * T(j)) / ((n_i(j) + 1) * num_neighbours);
        s2 = 0;
        for i = 1:uniq_classes_num
            if i ~= j
                continue;
            end
                s2 = s2 + delta_n(i, j) / (n_i(i) * num_neighbours);
        end
        predictions(j) = s1 - s2;
    end
    [~, index] = max(predictions);
    test_label(probe) = uniq_classes(index);
end