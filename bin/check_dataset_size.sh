#!/usr/bin/env bash
# Check the class distribution of train & val splits in a given dataset
# Usage: `./bin/check_dataset_size.sh path/to/dataset`
data_path=$1

train_pos=$(ls $data_path/train/mitosis/ | wc -l)
train_neg=$(ls $data_path/train/normal/ | wc -l)
val_pos=$(ls $data_path/val/mitosis/ | wc -l)
val_neg=$(ls $data_path/val/normal/ | wc -l)

echo "train mitoses: $train_pos"
echo "train normal: $train_neg"
echo "val mitoses: $val_pos"
echo "val normal: $val_neg"

