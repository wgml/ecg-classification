load('../../data/unique_labels_concat/label.txt');
cs = unique(label);
result = zeros(size(label));
for i = 1:length(cs)
    result = result + i * (label == cs(i));
end
figure
subplot(1,2,1)
histogram(result)
xlabel('Klasa')
ylabel('Liczność')
set(gca, 'YScale', 'log')
title('Podział na klasy')
result2 = (result == 1) + 2 * (result ~=1);
subplot(1,2,2)
C= categorical(result2, [1, 2], {'Normalny', 'Arytmia'});
histogram(C, 'BarWidth', 0.75)
ylabel('Liczność')
xlabel('Kategoria')
title('Podział binarny')