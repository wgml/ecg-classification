close all;
clear all;

%%loading data
%importdata = importdata('./ReferencyjneDane/101/AfterClustering.txt');
importdata = importdata('./ReferencyjneDane/101/ConvertedQRSRawData.txt');
load('./ReferencyjneDane/101/Class_IDs.txt');
data=importdata(:,2:18);

%%Normalization
for j = 1:size(data,2)
    vec = data(:,j);
    vec = vec - mean(vec);
    vec = vec/std(vec);
    data(:,j) = vec;
end

train_data_percentage = 0.7;
train_data_amount = floor(train_data_percentage * size(data,1));
train_data = data(1:train_data_amount,:);
test_data = data((train_data_amount+1):size(data,1),:);

      
K = 5;
[n,d] = knnsearch(train_data,test_data,'k',K,'distance','minkowski','p',5);
%[n,d] = knnsearch(train_data,test_data,'k',K,'distance','chebychev');

neighbours_labels = zeros(size(test_data,1),K);
label = zeros(2,size(test_data,1));

correct_detections = 0;
wrong_detections = 0;

for j=1:size(test_data,1)
    label(1,j)=Class_IDs(train_data_amount+j);
    counter1=0;
    counter2=0;
    counter3=0;
    
    for i=1:K
        neighbours_labels(j,i) = Class_IDs(n(j,i));
        if Class_IDs(n(j,i)) == 1
            counter1 = counter1 + 1;
        elseif Class_IDs(n(j,i)) == 2
            counter2 = counter2+1;
        else
            counter3 = counter3+1; 
        end
    end
        if counter1 > counter2 && counter1 > counter3
            label(2,j) = 1;
        elseif counter2 > counter1 && counter 2 > counter3
            label(2,j) = 2;
        else
            label(2,j) = 3;
        end      
   if label(1,j) == label(2,j)
       correct_detections = correct_detections+1;
   else
       wrong_detections = wrong_detections+1;
   end
end

correct_detections_percent = correct_detections/(correct_detections+wrong_detections)
