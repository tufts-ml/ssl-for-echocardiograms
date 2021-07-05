#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
CUDA_VISIBLE_DEVICES=$gpu_idx python ../../src/image_level/vat.py \
    --train_kimg $train_kimg \
    --train_dir $train_dir \
    --task_name $task_name \
    --train_labeled_files $train_labeled_files \
    --train_unlabeled_files $train_unlabeled_files \
    --valid_files $valid_files \
    --test_files $test_files \
    --class_weights $class_weights \
    --lr $lr \
    --wd $wd \
    --vat_eps $vat_eps \
    --report_type $report_type\
    