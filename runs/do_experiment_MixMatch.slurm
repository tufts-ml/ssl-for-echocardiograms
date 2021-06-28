#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
CUDA_VISIBLE_DEVICES=$gpu_idx python ../../src/image_level/mixmatch.py \
    --train_kimg $train_kimg \
    --train_dir $train_dir \
    --task_name $task_name \
    --dataset $dataset \
    --train_labeled_files $train_labeled_files \
    --train_unlabeled_files $train_unlabeled_files \
    --valid_files $valid_files \
    --test_files $test_files \
    --class_weights $class_weights \
    --lr $lr \
    --wd $wd \
    --beta $beta\
    --w_match $w_match\
    --scales $scales\
    --warmup_kimg $warmup_kimg\
    --warmup_delay $warmup_delay\
    --mixmode $mixmode \
    