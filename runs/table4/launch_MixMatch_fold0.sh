#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

export gpu_idx=2 
export ML_DATA="../../ML_DATA/TMED-156-52/fold0_multitask/64_MaintainingAspectRatio_ResizingThenPad_ExcludeDoppler_grayscale/"
export PYTHONPATH=$PYTHONPATH:.
export train_kimg=2000
export dataset="echo"
export class_weights="0.2300,0.7387,0.0313"
export lr=0.002
export wd=0.0002
export beta=0.75
export w_match=100.0
export warmup_delay=0
export warmup_kimg=1024
export scales=4
export train_dir="../../experiments/table4/MixMatch/fold0"
export task_name="ViewClassification"
export train_labeled_files='train-label_VIEW.tfrecord'
export train_unlabeled_files='train-unlabel_RU_VIEW.tfrecord,train-unlabel_PartiallyLabeled_VIEW.tfrecord'
export valid_files='valid_VIEW.tfrecord'
export test_files='test_VIEW.tfrecord'
export mixmode='xxy.yxy'

if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < ../do_experiment_MixMatch.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash ../do_experiment_MixMatch.slurm
fi
