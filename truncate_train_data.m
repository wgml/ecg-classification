function [data, labels] = truncate_train_data(data, labels, num_classes, max_diff_factor)
class_counts = histc(labels, 1:num_classes);

to_feed = min(class_counts(class_counts > 0));
for i=1:num_classes
    if class_counts(i) > 0
        class_counts(i) = to_feed;
    end
end
to_feed = sum(class_counts) * max_diff_factor;

data_new = zeros(to_feed, size(data,2));
labels_new = zeros(to_feed, 1);

new_idx = 1;
for i = 1:length(labels)
    c = labels(i);
    if class_counts(c) > 0
        data_new(new_idx, :) = data(i, :);
        labels_new(new_idx) = labels(i);
        
        new_idx = new_idx + 1;
        to_feed = to_feed - 1;
        class_counts(c) = class_counts(c) - 1;
    end
    
    if to_feed == 0
       break
   end
end
fprintf('Truncating train vector of %d elements into %d.', length(labels), length(labels_new));

data = data_new;
labels = labels_new;
end

