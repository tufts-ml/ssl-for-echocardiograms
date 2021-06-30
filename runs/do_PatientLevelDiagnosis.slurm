#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
python3 ../../src/patient_level/run_PatientLevelDiagnosis.py \
    --fold_idx $fold_idx \
    --split $split \
    --algo $algo \
    --View_predictions_dir $View_predictions_dir\
    --Diagnosis_predictions_dir $Diagnosis_predictions_dir\
    --Diagnosis_true_labels_dir $Diagnosis_true_labels_dir\
    --split_info_dir $split_info_dir\
    