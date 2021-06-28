#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
CUDA_VISIBLE_DEVICES=$gpu_idx python ../../src/image_level/fs_baseline.py \
    --load_ckpt $load_ckpt \
    --reset_global_step $reset_gloabl_step \
    --checkpoint_exclude_scopes $checkpoint_exclude_scopes \
    --trainable_scopes $trainable_scopes \
    --train_kimg $train_kimg \
    --train_dir $train_dir \
    --task_name $task_name \
    --dataset $dataset \
    --train_labeled_files $train_labeled_files \
    --valid_files $valid_files \
    --test_files $test_files \
    --class_weights $class_weights \
    --lr $lr \
    --wd $wd \
    --smoothing $smoothing \
    --scales $scales