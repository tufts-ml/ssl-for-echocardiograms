import os
import numpy as np
import sys
from easydict import EasyDict as edict

from utils import load_pickle, save_pickle, calculate_balanced_accuracy, generate_PatientOrder_DataIndicesRange_ImageCount

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fold_idx', default=0,type=int, help='which suggested split')
parser.add_argument('--split', default='test', type=str, help="test/val")
parser.add_argument('--algo', default='FS', type=str, help='looking at the image level predictions from which algo')
parser.add_argument('--View_predictions_dir', default='../../predictions/ImageLevel_predictions/')
parser.add_argument('--Diagnosis_predictions_dir', default='../../predictions/ImageLevel_predictions/')
parser.add_argument('--Diagnosis_true_labels_dir', default='../../split_info/TMED-156-52/')
parser.add_argument('--split_info_dir', default='../../split_info/TMED-156-52/')



def perform_PatientLevel_integration(args_dict):
    
    fold_idx = args_dict.fold_idx
    split = args_dict.split
    algo = args_dict.algo
    View_predictions_dir = args_dict.View_predictions_dir
    Diagnosis_predictions_dir = args_dict.Diagnosis_predictions_dir
    Diagnosis_true_labels_dir = args_dict.Diagnosis_true_labels_dir
    
    View_predictions = load_pickle(os.path.join(View_predictions_dir, 'fold{}'.format(fold_idx), split, algo), 'view_predictions.pkl')
    Diagnosis_predictions = load_pickle(os.path.join(Diagnosis_predictions_dir, 'fold{}'.format(fold_idx), split, algo), 'diagnosis_predictions.pkl')
    Diagnosis_true_labels = load_pickle(os.path.join(Diagnosis_true_labels_dir, 'fold{}'.format(fold_idx), split), '{}_diagnosis_labels.pkl'.format(split))

    #as sanity check:
    if fold_idx == 0:
        set_size = 5690
    elif fold_idx == 1:
        set_size = 5855
    elif fold_idx == 2:
        set_size = 5377
    elif fold_idx == 3:
        set_size = 5535
    
    
    total_images = Diagnosis_predictions.shape[0]
    assert total_images == set_size
             
    #get test_patients_order_list, test_patients_DataIndicesRange_list, test_patients_ImageCount_list
    patients_split_info_dir = args_dict.split_info_dir
    
    patients_order_list, patients_DataIndicesRange_list, patients_ImageCount_list = generate_PatientOrder_DataIndicesRange_ImageCount(patients_split_info_dir, fold_idx, split)

    
    #compared approaches
    
    #Take all available images of a patient, average the diagnosis predictions 
    SimpleAveraging_predicted_labels = []
    SimpleAveraging_predictions = []
    
    #Our reported prioritize view strategy
    ConfidencedBased_PrioritizedView_predicted_labels = []
    ConfidencedBased_PrioritizedView_predictions = []
    
    
    patient_true_diagnosis_labels = []
    
    #loop through each patient
    for idx, patient_id in enumerate(patients_order_list):
        print('Currently aggregating predictions for {}'.format(patient_id).center(100, '-'))
        this_patient_data_indices = list(range(total_images))[patients_DataIndicesRange_list[idx][0]:patients_DataIndicesRange_list[idx][1]] 
        
        this_patient_diagnosis_true_labels = Diagnosis_true_labels[this_patient_data_indices]
        
        assert len(list(set(this_patient_diagnosis_true_labels))) == 1, '1 patient can only have 1 diagnosis label'
        this_patient_diagnosis_single_label = this_patient_diagnosis_true_labels[0]
        
        #record this patient's true diagnosis label
        patient_true_diagnosis_labels.append(this_patient_diagnosis_single_label)
       
        this_patient_diagnosis_predictions = Diagnosis_predictions[this_patient_data_indices]
        this_patient_view_predictions = View_predictions[this_patient_data_indices]

        
        #SimpleAveraging:
        this_patient_SimpleAveraging_prediction = np.mean(this_patient_diagnosis_predictions, axis = 0) #a 1x3 vector
        this_patient_SimpleAveraging_predicted_label = np.argmax(this_patient_SimpleAveraging_prediction)
        
        #record this patient's SimpleAveraging predictions and predicted labels
        SimpleAveraging_predictions.append(this_patient_SimpleAveraging_prediction)
        SimpleAveraging_predicted_labels.append(this_patient_SimpleAveraging_predicted_label)
                        
#         print('true_diagnosis:{}, SoftMajorityVote predicted_diagnosis:{}\n'.format(this_patient_diagnosis_single_label, this_patient_SimpleAveraging_predicted_label))

        
        #ConfidenceBased_PrioritizedView
        this_patient_ViewRelevance =  np.sum(this_patient_view_predictions[:,:2], axis=1)
        this_patient_DiagnosisPrediction_with_ViewRelevance = np.mean(this_patient_diagnosis_predictions * this_patient_ViewRelevance[:, np.newaxis], axis=0)
        this_patient_DiagnosisPredictedLabel_with_ViewRelevance = np.argmax(this_patient_DiagnosisPrediction_with_ViewRelevance)

        #record this patient's ConfidenceBased_PrioritizedView predictions and predicted labels
        ConfidencedBased_PrioritizedView_predictions.append(this_patient_DiagnosisPrediction_with_ViewRelevance)
        ConfidencedBased_PrioritizedView_predicted_labels.append(this_patient_DiagnosisPredictedLabel_with_ViewRelevance)

#         print('true_diagnosis:{}, ConfidenceBased_PrioritizedView predicted_diagnosis:{}\n'.format(this_patient_diagnosis_single_label, this_patient_DiagnosisPredictedLabel_with_ViewRelevance))
        
#         print('\n')
        
    
    SimpleAveraging_balanced_accuracy = calculate_balanced_accuracy(patient_true_diagnosis_labels, SimpleAveraging_predicted_labels)
    ConfidenceBased_PrioritizedView_balanced_accuracy = calculate_balanced_accuracy(patient_true_diagnosis_labels, ConfidencedBased_PrioritizedView_predicted_labels)
   
                            
    returned_dict = edict()
    returned_dict.true_diagnosis_labels = patient_true_diagnosis_labels
    returned_dict.SimpleAveraging = {'balanced_accuracy': SimpleAveraging_balanced_accuracy, 'predicted_labels': np.array(SimpleAveraging_predicted_labels), 'predictions':np.array(SimpleAveraging_predictions)}
    returned_dict.ConfidenceBased_PrioritizedView = {'balanced_accuracy':ConfidenceBased_PrioritizedView_balanced_accuracy, 'predicted_labels':np.array(ConfidencedBased_PrioritizedView_predicted_labels), 'predictions':np.array(ConfidencedBased_PrioritizedView_predictions)}    
    

    return returned_dict

    
if __name__=='__main__':
    
    args = parser.parse_args()
    
    fold_idx = args.fold_idx
    split = args.split
    algo = args.algo
    View_predictions_dir = args.View_predictions_dir
    Diagnosis_predictions_dir = args.Diagnosis_predictions_dir
    Diagnosis_true_labels_dir = args.Diagnosis_true_labels_dir
    split_info_dir = args.split_info_dir
    
    
    input_args_dict=edict()
    input_args_dict.fold_idx = fold_idx
    input_args_dict.split= split
    input_args_dict.algo = algo
    input_args_dict.View_predictions_dir = View_predictions_dir
    input_args_dict.Diagnosis_predictions_dir = Diagnosis_predictions_dir
    input_args_dict.Diagnosis_true_labels_dir = Diagnosis_true_labels_dir
    input_args_dict.split_info_dir = split_info_dir
    
    returned_dict = perform_PatientLevel_integration(input_args_dict) 
    
    print('{}, fold{}, {}'.format(algo, fold_idx, split))
    print('SimpleAveraging_balanced_accuracy: {}'.format(returned_dict.SimpleAveraging['balanced_accuracy']))
    print('ConfidenceBased_PrioritizedView_balanced_accuracy: {}'.format(returned_dict.ConfidenceBased_PrioritizedView['balanced_accuracy']))

    



