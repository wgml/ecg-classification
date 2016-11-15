function [ result ] = n_closest_neighbours( train_data, classes, test_probe, n, dist_fcn )

train_amount = size(train_data, 1);

if nargin ~= 5
    p = 2;
    dist_fcn = @(x, y) sum(abs(x - y) .^ p) ^ (1 / p);
end
distances = zeros(size(train_amount, 1), 1);
for i = 1:train_amount
      distances(i) = dist_fcn(test_probe, train_data(i, :));
end

[~, idxs] = sort(distances);
result = classes(idxs(1:n));
