#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
CUDA_VISIBLE_DEVICES=$gpu_idx python ../../src/image_level/fs_multitask.py \
    --train_kimg $train_kimg \
    --train_dir $train_dir \
    --dataset $dataset \
    --train_labeled_files $train_labeled_files \
    --valid_files $valid_files \
    --test_files $test_files \
    --class_weights_diagnosis $class_weights_diagnosis \
    --class_weights_view $class_weights_view \
    --auxiliary_task_weight $auxiliary_task_weight \
    --lr $lr \
    --wd $wd \
    --smoothing $smoothing \
    --scales $scales \
    --multitask_network_flag $multitask_network_flag \
    
