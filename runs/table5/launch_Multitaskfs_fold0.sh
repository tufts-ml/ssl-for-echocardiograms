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
export ML_DATA="../../ML_DATA/TMED-18-18_MultitaskAblation/fold0_multitask/64_MaintainingAspectRatio_ResizingThenPad_ExcludeDoppler_grayscale/"
export PYTHONPATH=$PYTHONPATH:.
export train_kimg=2000
export dataset="echo"
export class_weights_diagnosis="0.3741,0.3541,0.2718"
export class_weights_view="0.2300,0.7387,0.0313"
export auxiliary_task_weight=3.0
export lr=0.0007
export wd=0.002
export smoothing=0.001
export scales=4
export train_dir="../../experiments/table5/FS_Multitask/fold0"
export task_name="DiagnosisClassification"
export train_labeled_files='train-label_multitask.tfrecord'
export valid_files='valid_multitask.tfrecord'
export test_files='test_multitask.tfrecord'
export multitask_network_flag=True


if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < ../do_experiment_MultitaskFS.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash ../do_experiment_MultitaskFS.slurm
fi

