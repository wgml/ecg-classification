function [ PredictionLabel ] = enn( train_data, classes, test_data, num_neighbours,dist_fcn )

train_amount = size(train_data, 1);

if nargin ~= 5
    dist_fcn = @(train, test) sqrt(sum(abs(ones(train_amount, 1) * test - train) .^ 2, 2));
end
 
uniq_classes = unique(classes);
uniq_classes = sort(uniq_classes,1,'ascend');
uniq_classes_num = length(uniq_classes);

T = zeros(uniq_classes_num);
n_i = zeros(uniq_classes_num, 1);

nn_distance = zeros(train_amount,num_neighbours);
nn_classes = zeros(train_amount,num_neighbours);

for idx = 1:train_amount
    distance = sqrt(sum(abs(ones(train_amount, 1) * train_data(idx,:) - train_data) .^ 2, 2));
    [sortDistance, sortDistanceIdx] = sort(distance);
    nn_distance(idx,:) = sortDistance(2:num_neighbours+1);
    nn_classes(idx,:) = classes(sortDistanceIdx(2:num_neighbours+1));
end

for class_idx = 1:uniq_classes_num
    class = uniq_classes(class_idx);
    class_idxs = find(classes == class);
    n_i(class_idx) = length(class_idxs);
    
    T(class_idx) = length(find(nn_classes(class_idxs,1:num_neighbours) == class_idx)) / (n_i(class_idx)*num_neighbours);
end

delta_n = zeros(uniq_classes_num);
for probe = 1:size(test_data, 1)
    test_probe = test_data(probe, :);
    
    distance = sqrt(sum(abs(ones(train_amount, 1) * test_probe - train_data) .^ 2, 2));
    [sortDistance, sortDistanceIdx] = sort(distance);
    x_nn_classes = classes(sortDistanceIdx(1:num_neighbours));

    k_i = zeros(uniq_classes_num, 1);
    for c_idx = 1:uniq_classes_num
       k_i(c_idx) = length(find(x_nn_classes == uniq_classes(c_idx)));
    end
    
    predictions = zeros(uniq_classes_num, 1);

    classes_num_ij = zeros(uniq_classes_num,1);
    classes_num_jj = zeros(uniq_classes_num,1);
    
    for j = 1:uniq_classes_num
       class_idx = j;
       test_label = uniq_classes(class_idx);
       class_idxs = find(classes == test_label);
       orig_nn_distance = nn_distance(class_idxs, num_neighbours);
       orig_nn_class = nn_classes(class_idxs, num_neighbours);
       
       dif_distance = distance(class_idxs,1) - orig_nn_distance;
       negative_dif=find(dif_distance <=0);
       classes_num_ij(j) = length(negative_dif);

       if(classes_num_ij(j) > 0)
           classes_num_jj(j) = length(find(orig_nn_class(negative_dif) == class_idx));
       end
    end
    
    for i = 1:uniq_classes_num
        delta_n_jj = classes_num_ij(i) - classes_num_jj(i);
        
        s1 = (delta_n_jj + k_i(i) - T(i) * num_neighbours) / ((n_i(i) + 1)*num_neighbours);
        s2 = sum(classes_num_jj ./ (n_i * num_neighbours)) - classes_num_jj(i) / (n_i(i) * num_neighbours);
        predictions(i) = s1 - s2;
    end
    
    [~, index] = max(predictions);
    PredictionLabel(probe) = uniq_classes(index);     
end
PredictionLabel = PredictionLabel';