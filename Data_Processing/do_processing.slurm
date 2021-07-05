#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
python data_processing.py \
    --result_save_root_dir $result_save_root_dir \
    --dataset_name $dataset_name \
    --fold $fold \
    --raw_data_dir $raw_data_dir \
    --suggested_split_file_dir $suggested_split_file_dir