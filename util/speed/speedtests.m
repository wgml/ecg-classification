close all;
clear all;

knn_data = importdata('knn_results.txt');
enn_data = importdata('enn_results.txt');

x_max = max(knn_data(:,1));
x = 1000:1000:x_max;
train_cut = [];
test_cut = [];
both_cut = [];
for i=1:size(knn_data,1)
    tr = knn_data(i, 1);
    te = knn_data(i, 2);
    va = knn_data(i, 3);
    
    if (tr == te)
        both_cut = [both_cut va];
    end
    if (te == x_max)
        train_cut = [train_cut va];
    end
    if (tr == x_max)
        test_cut = [test_cut va];
    end
end
figure
plot(x, both_cut, 'ro--', x, train_cut, 'go--', x, test_cut, 'bo--');
xlabel('Liczność ograniczonego zbioru')
ylabel('Czas [ms]')
grid on;
legend('Ograniczono oba zbiory', 'Ograniczony zbiór uczący', 'Ograniczony zbiór testowy');

x_max = max(enn_data(:,1));
x = 1000:1000:x_max;
train_cut = [];
test_cut = [];
both_cut = [];
for i=1:size(enn_data,1)
    tr = enn_data(i, 1);
    te = enn_data(i, 2);
    va = enn_data(i, 3);
    
    if (tr == te)
        both_cut = [both_cut va];
    end
    if (te == x_max)
        train_cut = [train_cut va];
    end
    if (tr == x_max)
        test_cut = [test_cut va];
    end
end
figure
plot(x, both_cut, 'ro--', x, train_cut, 'go--', x, test_cut, 'bo--');
xlabel('Liczność ograniczonego zbioru')
ylabel('Czas wykonania [ms]')
grid on;
legend('Ograniczono oba zbiory', 'Ograniczony zbiór uczący', 'Ograniczony zbiór testowy');