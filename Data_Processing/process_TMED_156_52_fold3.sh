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

export result_save_root_dir="../ML_DATA/" 
export dataset_name="TMED-156-52"
export fold="fold3"
export raw_data_dir="/cluster/tufts/hugheslab/zhuang12/MLHCData_Release/Release/20210609/images(Released)"
export suggested_split_file_dir="/cluster/tufts/hugheslab/zhuang12/MLHCData_Release/Release/20210609/SplitImageLabelMapping(Released)_WithMeta"
    

if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < do_processing.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash do_processing.slurm
fi
