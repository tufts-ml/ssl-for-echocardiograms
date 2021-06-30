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

export fold_idx=2
export split='test'
export algo='PretrainedMixMatch'
export View_predictions_dir='../../predictions/ImageLevel_predictions/'
export Diagnosis_predictions_dir='../../predictions/ImageLevel_predictions/'
export Diagnosis_true_labels_dir='../../split_info/TMED-156-52/'
export split_info_dir='../../split_info/TMED-156-52/'



## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < ../do_PatientLevelDiagnosis.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash ../do_PatientLevelDiagnosis.slurm
fi



