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
export ML_DATA="../../ML_DATA/TMED-18-18/fold1/"
export PYTHONPATH=$PYTHONPATH:.
export train_kimg=2000
export class_weights="0.3406,0.3159,0.3435"
export lr=0.0007
export wd=0.0002
export train_dir="../../experiments/table5/PretrainedFS/fold1"
export task_name="DiagnosisClassification"
export report_type="EMA_BalancedAccuracy"
export train_labeled_files='train-label_DIAGNOSIS.tfrecord'
export valid_files='valid_DIAGNOSIS.tfrecord'
export test_files='test_DIAGNOSIS.tfrecord'
export reset_global_step=True
export checkpoint_exclude_scopes="dense"
export trainable_scopes="None"
export load_ckpt="../../models/TMED-156-52/fold0_view/best_validation_balanced_accuracy.ckpt" #just a demo


if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < ../do_experiment_PretrainedFS.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash ../do_experiment_PretrainedFS.slurm
fi

