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
export ML_DATA="../../ML_DATA/TMED-156-52/fold2/"
export PYTHONPATH=$PYTHONPATH:.
export train_kimg=2000
export class_weights="0.3876,0.3561,0.2564"
export lr=0.002
export wd=0.0002
export w_match=100.0
export warmup_delay=500
export warmup_kimg=500
export train_dir="../../experiments/table6/MixMatch/fold2"
export task_name="DiagnosisClassification"
export report_type="EMA_BalancedAccuracy"
export train_labeled_files='train-label_DIAGNOSIS.tfrecord'
export train_unlabeled_files='train-unlabel_RU_DIAGNOSIS.tfrecord,train-unlabel_PartiallyLabeled_DIAGNOSIS.tfrecord'
export valid_files='valid_DIAGNOSIS.tfrecord'
export test_files='test_DIAGNOSIS.tfrecord'
export mixmode='xxy.yxy'

if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < ../do_experiment_MixMatch.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash ../do_experiment_MixMatch.slurm
fi
