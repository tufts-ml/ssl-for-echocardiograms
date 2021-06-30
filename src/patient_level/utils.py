import os
import pickle
import numpy as np

from sklearn.metrics import confusion_matrix as sklearn_cm

#some helper functions
def generate_PatientOrder_DataIndicesRange_ImageCount(split_info_dir, fold_idx, split='test'):
    
    #load the patient_order_list and patient_level_count_dicts pre-generated when processing data
    
    patients_order_list = load_pickle(os.path.join(split_info_dir, 'fold{}'.format(fold_idx), split), '{}_patient_order_list.pkl'.format(split))
    patientlevel_count_dicts = load_pickle(os.path.join(split_info_dir, 'fold{}'.format(fold_idx), split), '{}_patient_level_count_dicts.pkl'.format(split))

    
    num_patients = len(patients_order_list)
    patients_ImageCount_list = []
    patients_DataIndicesRange_list = []
    
    for patient_id in patients_order_list:
        this_patient_number_images = 0
        for view, view_labels in patientlevel_count_dicts[patient_id]['view_labels_count'].items():
            this_patient_number_images += view_labels
        
        patients_ImageCount_list.append(this_patient_number_images)
    
    
    patient_DataIndicesEndpoints_list = np.insert(np.cumsum(patients_ImageCount_list), 0, 0)
    
    for i in range(num_patients):
        patients_DataIndicesRange_list.append((patient_DataIndicesEndpoints_list[i], patient_DataIndicesEndpoints_list[i+1]))
        
    
    return patients_order_list, patients_DataIndicesRange_list, patients_ImageCount_list

def load_pickle(result_dir, filename):
    with open(os.path.join(result_dir, filename), 'rb') as f:
        data = pickle.load(f)
    
    return data


def save_pickle(save_dir, save_file_name, data):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data_save_fullpath = os.path.join(save_dir, save_file_name)
    with open(data_save_fullpath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

def calculate_balanced_accuracy(true_labels, predictions, return_type = 'balanced_accuracy'):
    '''
    used particularly for this 3-classes classification task
    '''
    
    confusion_matrix = sklearn_cm(true_labels, predictions)
    
    class0_recall = confusion_matrix[0,0]/np.sum(confusion_matrix[0])
    class1_recall = confusion_matrix[1,1]/np.sum(confusion_matrix[1])
    class2_recall = confusion_matrix[2,2]/np.sum(confusion_matrix[2])
    
    balanced_accuracy = (1/3)*class0_recall + (1/3)*class1_recall + (1/3)*class2_recall
    
    if return_type == 'all':
        return balanced_accuracy * 100, class0_recall * 100, class1_recall * 100, class2_recall * 100
    elif return_type == 'balanced_accuracy':
        return balanced_accuracy * 100
    else:
        raise NameError('Unsupported return_type in this calculate_balanced_accuracy fn')
