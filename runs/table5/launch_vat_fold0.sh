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
export ML_DATA="../../ML_DATA/TMED-18-18/fold0/"
export PYTHONPATH=$PYTHONPATH:.
export train_kimg=2000
export class_weights="0.3385,0.3292,0.3323"
export lr=0.0007
export wd=0.0002
export vat_eps=6.0
export train_dir="../../experiments/table5/vat/fold0"
export task_name="DiagnosisClassification"
export report_type="EMA_BalancedAccuracy"
export train_labeled_files='train-label_DIAGNOSIS.tfrecord'
export train_unlabeled_files='train-unlabel_DIAGNOSIS.tfrecord'
export valid_files='valid_DIAGNOSIS.tfrecord'
export test_files='test_DIAGNOSIS.tfrecord'

if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < ../do_experiment_vat.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash ../do_experiment_vat.slurm
fi
