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

export gpu_idx=0 
export ML_DATA="../../ML_DATA/TMED-156-52/fold2/"
export PYTHONPATH=$PYTHONPATH:.
export train_kimg=2000
export class_weights="0.2542,0.7123,0.0335"
export lr=0.002
export wd=0.0002
export w_match=0.0
export warmup_delay=0
export warmup_kimg=1024
export train_dir="../../experiments/table4/AugmentOnlyMixMatch/fold2"
export task_name="ViewClassification"
export report_type="RAW_BalancedAccuracy"
export train_labeled_files='train-label_VIEW.tfrecord'
export train_unlabeled_files='train-unlabel_RU_VIEW.tfrecord,train-unlabel_PartiallyLabeled_VIEW.tfrecord'
export valid_files='valid_VIEW.tfrecord'
export test_files='test_VIEW.tfrecord'
export mixmode='xxy.'

if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < ../do_experiment_MixMatch.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash ../do_experiment_MixMatch.slurm
fi
