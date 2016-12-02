function [ result ] = n_closest_neighbours( train_data, classes, test_probe, n, dist_fcn )

train_amount = size(train_data, 1);

if nargin ~= 5
    dist_fcn = @(train, test) sqrt(sum(abs(ones(train_amount, 1) * test - train) .^ 2, 2)));
end
distances = zeros(size(train_amount, 1), 1);

distances = dist_fcn(train_data, test_probe);

[~, idxs] = sort(distances);
result = classes(idxs(1:n));