
files = dir('../data/data_reused_labels/*');
files = files(3:end); %remove . and ..

str = '';
for i=1:38
    file = files(i, 1).name;
 
    str = strvcat(str,strcat(file, ' & ', num2str(sensitivity(i,1)), ' & ', num2str(specificity(i,1)), ' & ', num2str(sensitivity(i,2)), ' & ', num2str(specificity(i,2)), '\\'));
    str = strvcat(str, '\hline');
end

fileID = fopen('str.txt','w');
fprintf(fileID, '%s', str);
fclose(fileID);