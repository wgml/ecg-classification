function [ result ] = knn( train_data, classes, test_data, num_neighbours, dist_fcn )

train_amount = size(train_data, 1);
test_amount = size(test_data, 1);
result = zeros(test_amount, 1);

if nargin ~= 5
    dist_fcn = @(train, test) sqrt(sum(abs(ones(train_amount, 1) * test - train) .^ 2, 2));
end

for probe_idx = 1:test_amount
    test_probe = test_data(probe_idx, :);
    distances = zeros(size(train_amount, 1), 1);
    distances = dist_fcn(train_data, test_probe);
    % get closest
    [~, idxs] = sort(distances);
    result(probe_idx) = mode(classes(idxs(1:num_neighbours)));
end