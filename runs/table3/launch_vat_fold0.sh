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

export gpu_idx=1
export ML_DATA="../../ML_DATA/TMED-18-18/fold0/"
export PYTHONPATH=$PYTHONPATH:.
export train_kimg=2000
export class_weights="0.2447,0.7238,0.0316"
export lr=0.0007
export wd=0.002
export vat_eps=2.0
export train_dir="../../experiments/table3/vat/fold0"
export task_name="ViewClassification"
export report_type="RAW_BalancedAccuracy"
export train_labeled_files='train-label_VIEW.tfrecord'
export train_unlabeled_files='train-unlabel_VIEW.tfrecord'
export valid_files='valid_VIEW.tfrecord'
export test_files='test_VIEW.tfrecord'

if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < ../do_experiment_vat.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash ../do_experiment_vat.slurm
fi
