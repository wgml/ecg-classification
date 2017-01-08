#!/bin/sh
rm -rf unique_labels_concat
python ../util/convert_annotations_generic.py data_unique_labels > data_unique_labels.log
python ../util/convert_annotations.py data_reused_labels > data_reused_labels.log

for f in data_unique_labels/*;
do
        split -l $[ $(wc -l $f/data.txt|cut -d" " -f1) * 70 / 100 ] $f/data.txt
        mv xaa $f/train_data.txt
        mv xab $f/test_data.txt
        split -l $[ $(wc -l $f/label.txt|cut -d" " -f1) * 70 / 100 ] $f/label.txt
        mv xaa $f/train_label.txt
        mv xab $f/test_label.txt;
done

for f in data_reused_labels/*;
do
        split -l $[ $(wc -l $f/data.txt|cut -d" " -f1) * 70 / 100 ] $f/data.txt
        mv xaa $f/train_data.txt
        mv xab $f/test_data.txt
        split -l $[ $(wc -l $f/label.txt|cut -d" " -f1) * 70 / 100 ] $f/label.txt
        mv xaa $f/train_label.txt
        mv xab $f/test_label.txt;
done

mkdir unique_labels_concat
cat data_unique_labels/*/test_data.txt > unique_labels_concat/test_data.txt
cat data_unique_labels/*/train_data.txt > unique_labels_concat/train_data.txt
cat data_unique_labels/*/train_label.txt > unique_labels_concat/train_label.txt
cat data_unique_labels/*/test_label.txt > unique_labels_concat/test_label.txt
