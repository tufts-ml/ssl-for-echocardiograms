import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sklearn_cm
import seaborn as sns

from libml.utils import calculate_accuracy, calculate_balanced_accuracy, load_pickle


def get_percentile_curve(bootstrap_accuracy_curve_list, upper_percentile, lower_percentile):
    
    bootstrap_accuracy_curve_array = np.array(bootstrap_accuracy_curve_list)

    upper_percentile_curve = np.percentile(bootstrap_accuracy_curve_array, upper_percentile, axis = 0)
    lower_percentile_curve = np.percentile(bootstrap_accuracy_curve_array, lower_percentile, axis = 0)

    return upper_percentile_curve, lower_percentile_curve 


def plot_confusion_matrix(data, labels, output_filename, normalized_option = None):
    """Plot confusion matrix using heatmap.
 
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
 
    """
    print('Inside plot_confusion_matrix, data is {}'.format(data), flush = True)
    sns.set(color_codes=True)
    plt.figure(1, figsize=(8, 5))
 
    plt.title("Confusion Matrix")
    
    sns.set(font_scale=1.4)
    if normalized_option == 'Recall':
        data = data.astype(np.float16)
        data[0] = data[0]/np.sum(data[0])
        data[1] = data[1]/np.sum(data[1])
        data[2] = data[2]/np.sum(data[2])
        ax = sns.heatmap(data, annot=True, 
            fmt='.01%', cmap='Blues')
    
        
    elif normalized_option == 'Error':
        data = data.astype(np.float16)
        np.fill_diagonal(data, 0)
        data = data/np.sum(data)
        ax = sns.heatmap(data, annot=True, 
            fmt='.01%', cmap='Blues')
 
    else:
        ax = sns.heatmap(data, annot=True, 
            fmt='d', cmap='Blues')
    
    #ax = sns.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, annot_kws={'size':16})
    
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
 
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    ax.set_ylim([3, 0])

    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    #plt.show()
    plt.close()
    

def retrieve_predictions(predictions_dir, predictions_file_list, partition):
    #caching train predictions
    ema_predictions_list = []
    raw_predictions_list = []
    labels = []
    
    base_true_labels = None
    base_length = None #base_length is the size of the dataset, at each evaluation epoch, ensure the dataset is the same
    for idx, predictions_file in enumerate(predictions_file_list):
        if partition == 'train_labeled':
            print('currently processing {} predictions at step: {}'.format(partition, predictions_file.split('_')[3]))
        else:
            print('currently processing {} predictions at step: {}'.format(partition, predictions_file.split('_')[2]))

        data = load_pickle(predictions_dir, predictions_file)
        ema_predictions = data['ema_predictions']
        raw_predictions = data['raw_predictions']
        true_labels = data['true_labels']
        
        if base_length is None:
            base_length = len(ema_predictions)
        else:
            assert base_length == len(ema_predictions)
            
        if base_true_labels is None:
            base_true_labels = true_labels
        else:
            assert (base_true_labels == true_labels).all(), 'file at {} is problematic'.format(os.path.join(predictions_dir, predictions_file))
        
        ema_predictions_list.append(ema_predictions)
        raw_predictions_list.append(raw_predictions)
        labels.append(true_labels)
        
    return labels, ema_predictions_list, raw_predictions_list


    
def write_stats(valid_accuracy_curve, test_accuracy_curve, test_predictions_file_list, file_writer, predictions_dir, result_save_dir, report_type, task_name):
    
    '''
    Inside this function:
    "valid_accuracy_curve", "test_accuracy_curve", "max_valid_accuracy", "max_valid_accuracy_epoch", "test_accuracy_at_max_valid_epoch", "test_predictions_at_max_val_accuracy"
    
    generally refer to one of the RAW_Accuracy, RAW_BalancedAccuracy, EMA_Accuracy, EMA_BalancedAccuracy case. 
    '''    
    
    if report_type == 'RAW_BalancedAccuracy':
        output_string1 = 'max RAW valid balanced accuracy is {}, at epoch {}\n'
        output_string2 = 'At max RAW valid balanced accuracy epoch, RAW test balanced accuracy is {}\n\n'
        predictions_name = 'raw_predictions'
        recorded_accuracy_name = 'raw_balanced_accuracy'
        accuracy_calculation_method = calculate_balanced_accuracy
        
        
    elif report_type == 'EMA_BalancedAccuracy':
        output_string1 = 'max EMA valid balanced accuracy is {}, at epoch {}\n'
        output_string2 = 'At max EMA valid balanced accuracy epoch, EMA test balanced accuracy is {}\n\n'
        predictions_name = 'ema_predictions'
        recorded_accuracy_name = 'ema_balanced_accuracy'
        accuracy_calculation_method = calculate_balanced_accuracy
        
    else:
        raise NameError('Unsupported report type (write_stats)')
    
    
    
    max_valid_accuracy = np.max(valid_accuracy_curve)
    max_valid_accuracy_epoch = np.argmax(valid_accuracy_curve)
    test_accuracy_at_max_valid_epoch = test_accuracy_curve[max_valid_accuracy_epoch]
    
    print(output_string1.format(max_valid_accuracy, max_valid_accuracy_epoch))
    print(output_string2.format(test_accuracy_at_max_valid_epoch))
#     print('max valid accuracy is {}, at epoch {}\n'.format(max_valid_accuracy, max_valid_accuracy_epoch))
#     print('At max valid accuracy epoch: test accuracy is {} \n\n'.format(test_accuracy_at_max_valid_epoch))
    
    test_predictions_at_max_val_accuracy = load_pickle(predictions_dir, test_predictions_file_list[max_valid_accuracy_epoch])
    
    test_confusion_matrix = sklearn_cm(test_predictions_at_max_val_accuracy['true_labels'], test_predictions_at_max_val_accuracy[predictions_name].argmax(1))
    
    
    if task_name == 'ViewClassification':
        confusion_matrix_figure_label = ['PLAX', 'PSAX AoV', 'Other']
    elif task_name == 'DiagnosisClassification':
        confusion_matrix_figure_label = ['No_as', 'Mild/Mod_as', 'Severe_as']
    else:
        raise NameError('Unsupported task name (write_stats)')
    
    plot_confusion_matrix(test_confusion_matrix, confusion_matrix_figure_label, os.path.join(result_save_dir, 'test_confusion_matrix_at_max_validation_criterion.png'))
    plot_confusion_matrix(test_confusion_matrix, confusion_matrix_figure_label, os.path.join(result_save_dir, 'test_confusion_matrix_at_max_validation_criterion_RecallNormalized.png'), normalized_option = 'Recall')
    plot_confusion_matrix(test_confusion_matrix, confusion_matrix_figure_label, os.path.join(result_save_dir, 'test_confusion_matrix_at_max_validation_criterion_ErrorNormalized.png'), normalized_option = 'Error')

    
    #write to txt file using file_writer
    file_writer.write(output_string1.format(max_valid_accuracy, max_valid_accuracy_epoch))
    file_writer.write(output_string2.format(test_accuracy_at_max_valid_epoch))



def generate_bootstrap_accuracy_curve(original_predictions, original_labels, num_bootstrap_samples):
    rng = np.random.RandomState(0)
    
    original_accuracy_curve = [] #each element is the accuracy at that step
    original_balanced_accuracy_curve = [] #each element is the balanced accuracy at that step
    
    for j in range(len(original_predictions)): #len(original_predictions) is how many steps
        accuracy = calculate_accuracy(original_labels[j], original_predictions[j].argmax(1))
        balanced_accuracy = calculate_balanced_accuracy(original_labels[j], original_predictions[j].argmax(1))
        
        original_accuracy_curve.append(accuracy)
        original_balanced_accuracy_curve.append(balanced_accuracy)
    
    bootstrap_accuracy_curve_list = [] #each element is the accuracy curve for one bootstrap sample
    bootstrap_balanced_accuracy_curve_list = [] #each element is the balanced accuracy curve for one bootstrap sample
    
    for i in range(num_bootstrap_samples):
        ix = np.array(range(len(original_predictions[0]))) #the size of the dataset at each evaluation step are the same
        bootstrap_ix = rng.choice(ix, len(original_predictions[0]), replace = True)
#         print('i is {}, bootstrap_ix is {}'.format(i, bootstrap_ix))

        bootstrap_predictions = [i[bootstrap_ix] for i in original_predictions] #a list: each element is the predictions at a step
        bootstrap_true_labels = [i[bootstrap_ix] for i in original_labels]
        
        accuracy_curve_this_bootstrap_sample = []
        balanced_accuracy_curve_this_bootstrap_sample = []
        
        for j in range(len(bootstrap_predictions)): #each element is the predictions at one step
            #accuracy = (bootstrap_predictions[j].argmax(1) == bootstrap_true_labels[j]).mean() * 100
            
            accuracy = calculate_accuracy(bootstrap_true_labels[j], bootstrap_predictions[j].argmax(1))
            balanced_accuracy = calculate_balanced_accuracy(bootstrap_true_labels[j], bootstrap_predictions[j].argmax(1))
            
            accuracy_curve_this_bootstrap_sample.append(accuracy)
            balanced_accuracy_curve_this_bootstrap_sample.append(balanced_accuracy)
            
        bootstrap_accuracy_curve_list.append(accuracy_curve_this_bootstrap_sample)
        bootstrap_balanced_accuracy_curve_list.append(balanced_accuracy_curve_this_bootstrap_sample)
    
    return original_accuracy_curve, bootstrap_accuracy_curve_list, original_balanced_accuracy_curve, bootstrap_balanced_accuracy_curve_list 
        
    
    
def save_diagnosis_plots(figure_title, result_save_dir, test_original_accuracy_curve, test_lower_percentile_curve, test_upper_percentile_curve, valid_original_accuracy_curve, valid_lower_percentile_curve, valid_upper_percentile_curve, train_labeled_losses, train_unlabeled_losses_unscaled, train_unlabeled_losses_scaled, ylim_lower, ylim_upper, report_type):
    
    
    if report_type == 'RAW_BalancedAccuracy':
        test_original_accuracy_curve_label = 'RAW_test_balanced_accuracy'
        valid_original_accuracy_curve_label = 'RAW_valid_balanced_accuracy'   
    
    elif report_type == 'EMA_BalancedAccuracy':
        test_original_accuracy_curve_label = 'EMA_test_balanced_accuracy'
        valid_original_accuracy_curve_label = 'EMA_valid_balanced_accuracy'
    else:
        raise NameError('Unsupported report type')


    fig = plt.figure(figsize=(15,8))
    fig.suptitle(figure_title, fontsize=14, fontweight='bold')

    ax_1 = fig.add_subplot(2,2,1)
    ax_2 = fig.add_subplot(2,2,2, sharey = ax_1, sharex = ax_1)

    ax_3 = fig.add_subplot(2,2,3)
    ax_4 = fig.add_subplot(2,2,4, sharex = ax_3)

    ax_1.plot(list(range(len(test_original_accuracy_curve))), test_original_accuracy_curve, label = test_original_accuracy_curve_label)
    ax_1.fill_between(list(range(len(test_lower_percentile_curve))), test_lower_percentile_curve, test_upper_percentile_curve, alpha=0.5, color = 'red')
    ax_2.plot(list(range(len(valid_original_accuracy_curve))), valid_original_accuracy_curve, label = valid_original_accuracy_curve_label)
    ax_2.fill_between(list(range(len(valid_lower_percentile_curve))), valid_lower_percentile_curve, valid_upper_percentile_curve, alpha=0.5, color = 'red')

    ax_3.plot(list(range(len(train_labeled_losses))), train_labeled_losses, label = 'labeled_loss')
    ax_4.plot(list(range(len(train_unlabeled_losses_unscaled))), train_unlabeled_losses_unscaled, label = 'unlabeled_loss_unscaled')
    ax_4.plot(list(range(len(train_unlabeled_losses_scaled))), train_unlabeled_losses_scaled, label = 'unlabeled_loss_scaled')
    
    ax_1.set_ylim([ylim_lower, ylim_upper])
    ax_1.legend()
    ax_2.legend()
    ax_3.legend()
    ax_4.legend()
    
    figure_save_path = os.path.join(result_save_dir, figure_title + '_training_curves.png')
    plt.savefig(figure_save_path)
    plt.close()
    #plt.show()



def save_diagnosis_plots_multitask(figure_title, result_save_dir, test_original_accuracy_curve, test_lower_percentile_curve, test_upper_percentile_curve, valid_original_accuracy_curve, valid_lower_percentile_curve, valid_upper_percentile_curve, total_loss, unweighted_diagnosis_loss, weighted_diagnosis_loss, unweighted_view_loss, weighted_view_loss, scaled_weighted_view_loss, ylim_lower, ylim_upper, report_type):
    
        
    if report_type == 'RAW_BalancedAccuracy':
        test_original_accuracy_curve_label = 'RAW_test_balanced_accuracy'
        valid_original_accuracy_curve_label = 'RAW_valid_balanced_accuracy'   
        
    elif report_type == 'EMA_BalancedAccuracy':
        test_original_accuracy_curve_label = 'EMA_test_balanced_accuracy'
        valid_original_accuracy_curve_label = 'EMA_valid_balanced_accuracy'
    else:
        raise NameError('Unsupported report type')


    fig = plt.figure(figsize=(15,8))
    fig.suptitle(figure_title, fontsize=14, fontweight='bold')

    ax_1 = fig.add_subplot(2,2,1)
    ax_2 = fig.add_subplot(2,2,2, sharey = ax_1, sharex = ax_1)

    ax_3 = fig.add_subplot(2,2,3)
    ax_4 = fig.add_subplot(2,2,4, sharex = ax_3)

    ax_1.plot(list(range(len(test_original_accuracy_curve))), test_original_accuracy_curve, label = test_original_accuracy_curve_label)
    ax_1.fill_between(list(range(len(test_lower_percentile_curve))), test_lower_percentile_curve, test_upper_percentile_curve, alpha=0.5, color = 'red')
    ax_2.plot(list(range(len(valid_original_accuracy_curve))), valid_original_accuracy_curve, label = valid_original_accuracy_curve_label)
    ax_2.fill_between(list(range(len(valid_lower_percentile_curve))), valid_lower_percentile_curve, valid_upper_percentile_curve, alpha=0.5, color = 'red')

    ax_3.plot(list(range(len(total_loss))), total_loss, label='total loss')
    ax_3.plot(list(range(len(weighted_diagnosis_loss))), weighted_diagnosis_loss, label='weighted diagnosis loss')
    ax_3.plot(list(range(len(weighted_view_loss))), weighted_view_loss, label='weighted view loss')
    ax_3.plot(list(range(len(scaled_weighted_view_loss))), scaled_weighted_view_loss, label='scaled_weighted_view_loss')
    

    ax_1.set_ylim([ylim_lower, ylim_upper])
    ax_1.legend()
    ax_2.legend()
    ax_3.legend()
    
    figure_save_path = os.path.join(result_save_dir, figure_title + '_training_curves.png')
    plt.savefig(figure_save_path)
    plt.close()
    #plt.show()

    
def perform_analysis(figure_title, experiment_dir, result_save_dir, num_bootstrap_samples, bootstrap_upper_percentile, bootstrap_lower_percentile, ylim_lower, ylim_upper, report_type, task_name):
    
    #crate file writer object
    file_writer = open(os.path.join(result_save_dir, 'accuarcy_writer_sanitycheck.txt'), 'w')
    
    losses_dir = os.path.join(experiment_dir, 'losses')
    predictions_dir = os.path.join(experiment_dir, 'predictions')
    
    #load train, val, test accuracy at each epoch
    losses_dict = load_pickle(losses_dir, 'losses_dict.pkl')
    labeled_loss = losses_dict['labeled_losses']
    unlabeled_loss_unscaled = losses_dict['unlabeled_losses_unscaled']
    unlabeled_loss_scaled = losses_dict['unlabeled_losses_scaled']


    #load predictions at each epoch
    train_predictions_file_list = sorted([file for file in os.listdir(predictions_dir) if file.startswith('train_labeled')], key = lambda s: int(s.split('_')[3]))
    valid_predictions_file_list = sorted([file for file in os.listdir(predictions_dir) if file.startswith('valid')], key = lambda s: int(s.split('_')[2]))
    test_predictions_file_list = sorted([file for file in os.listdir(predictions_dir) if file.startswith('test')], key = lambda s: int(s.split('_')[2]))

    
    #caching the predictions at each epoch
    train_labels, train_ema_predictions, train_raw_predictions = retrieve_predictions(predictions_dir, train_predictions_file_list, 'train_labeled')
    valid_labels, valid_ema_predictions, valid_raw_predictions = retrieve_predictions(predictions_dir, valid_predictions_file_list, 'valid')
    test_labels, test_ema_predictions, test_raw_predictions = retrieve_predictions(predictions_dir, test_predictions_file_list, 'test')
    
    
    assert len(train_ema_predictions) == len(valid_ema_predictions) == len(test_ema_predictions) == len(train_raw_predictions) == len(valid_raw_predictions) == len(test_raw_predictions)

    #report type selection:
    
    if report_type == 'RAW_BalancedAccuracy':
        _, _, train_original_accuracy_curve, train_bootstrap_accuracy_curve_list = generate_bootstrap_accuracy_curve(train_raw_predictions, train_labels, num_bootstrap_samples)
    
        _, _, valid_original_accuracy_curve, valid_bootstrap_accuracy_curve_list = generate_bootstrap_accuracy_curve(valid_raw_predictions, valid_labels, num_bootstrap_samples)

        _, _, test_original_accuracy_curve, test_bootstrap_accuracy_curve_list = generate_bootstrap_accuracy_curve(test_raw_predictions, test_labels, num_bootstrap_samples)
    
    
    elif report_type == 'EMA_BalancedAccuracy':
        _, _, train_original_accuracy_curve, train_bootstrap_accuracy_curve_list = generate_bootstrap_accuracy_curve(train_ema_predictions, train_labels, num_bootstrap_samples)
    
        _, _, valid_original_accuracy_curve, valid_bootstrap_accuracy_curve_list = generate_bootstrap_accuracy_curve(valid_ema_predictions, valid_labels, num_bootstrap_samples)

        _, _, test_original_accuracy_curve, test_bootstrap_accuracy_curve_list = generate_bootstrap_accuracy_curve(test_ema_predictions, test_labels, num_bootstrap_samples)
    else:
        raise NameError('Unsupported report type')
        
    
    
    #write the max validation criterion and the corresponding test criterion to file                      
    write_stats(valid_original_accuracy_curve, test_original_accuracy_curve, test_predictions_file_list, file_writer, predictions_dir, result_save_dir, report_type, task_name)
                      
    
    train_upper_percentile_curve, train_lower_percentile_curve = get_percentile_curve(train_bootstrap_accuracy_curve_list, bootstrap_upper_percentile, bootstrap_lower_percentile)
    
    valid_upper_percentile_curve, valid_lower_percentile_curve = get_percentile_curve(valid_bootstrap_accuracy_curve_list, bootstrap_upper_percentile, bootstrap_lower_percentile)
    
    test_upper_percentile_curve, test_lower_percentile_curve = get_percentile_curve(test_bootstrap_accuracy_curve_list, bootstrap_upper_percentile, bootstrap_lower_percentile)
    
    
    save_diagnosis_plots(figure_title, result_save_dir, test_original_accuracy_curve, test_lower_percentile_curve, test_upper_percentile_curve, valid_original_accuracy_curve, valid_lower_percentile_curve, valid_upper_percentile_curve, labeled_loss, unlabeled_loss_unscaled, unlabeled_loss_scaled, ylim_lower, ylim_upper, report_type)


def perform_analysis_multitask(figure_title, experiment_dir, result_save_dir, num_bootstrap_samples, bootstrap_upper_percentile, bootstrap_lower_percentile, ylim_lower, ylim_upper, report_type, task_name):
    
    #crate file writer object
    file_writer = open(os.path.join(result_save_dir, 'accuarcy_writer_sanitycheck.txt'), 'w')
    
    losses_dir = os.path.join(experiment_dir, 'losses')
    predictions_dir = os.path.join(experiment_dir, 'predictions')
    
    #load train, val, test accuracy at each epoch
    losses_dict = load_pickle(losses_dir, 'losses_dict.pkl')
    total_loss = losses_dict['total_loss']
    unweighted_diagnosis_loss = losses_dict['unweighted_diagnosis_loss']
    weighted_diagnosis_loss = losses_dict['weighted_diagnosis_loss']
    unweighted_view_loss = losses_dict['unweighted_view_loss']
    weighted_view_loss = losses_dict['weighted_view_loss']
    scaled_weighted_view_loss = losses_dict['scaled_weighted_view_loss']

    #load predictions at each epoch
    train_predictions_file_list = sorted([file for file in os.listdir(predictions_dir) if file.startswith('train_labeled')], key = lambda s: int(s.split('_')[3]))
    valid_predictions_file_list = sorted([file for file in os.listdir(predictions_dir) if file.startswith('valid')], key = lambda s: int(s.split('_')[2]))
    test_predictions_file_list = sorted([file for file in os.listdir(predictions_dir) if file.startswith('test')], key = lambda s: int(s.split('_')[2]))

    
    #caching the predictions at each epoch
    train_labels, train_ema_predictions, train_raw_predictions = retrieve_predictions(predictions_dir, train_predictions_file_list, 'train_labeled')
    valid_labels, valid_ema_predictions, valid_raw_predictions = retrieve_predictions(predictions_dir, valid_predictions_file_list, 'valid')
    test_labels, test_ema_predictions, test_raw_predictions = retrieve_predictions(predictions_dir, test_predictions_file_list, 'test')
    
    
    assert len(train_ema_predictions) == len(valid_ema_predictions) == len(test_ema_predictions) == len(train_raw_predictions) == len(valid_raw_predictions) == len(test_raw_predictions)

    #report type selection:
    
    if report_type == 'RAW_BalancedAccuracy':
        _, _, train_original_accuracy_curve, train_bootstrap_accuracy_curve_list = generate_bootstrap_accuracy_curve(train_raw_predictions, train_labels, num_bootstrap_samples)
    
        _, _, valid_original_accuracy_curve, valid_bootstrap_accuracy_curve_list = generate_bootstrap_accuracy_curve(valid_raw_predictions, valid_labels, num_bootstrap_samples)

        _, _, test_original_accuracy_curve, test_bootstrap_accuracy_curve_list = generate_bootstrap_accuracy_curve(test_raw_predictions, test_labels, num_bootstrap_samples)
    
    
    elif report_type == 'EMA_BalancedAccuracy':
        _, _, train_original_accuracy_curve, train_bootstrap_accuracy_curve_list = generate_bootstrap_accuracy_curve(train_ema_predictions, train_labels, num_bootstrap_samples)
    
        _, _, valid_original_accuracy_curve, valid_bootstrap_accuracy_curve_list = generate_bootstrap_accuracy_curve(valid_ema_predictions, valid_labels, num_bootstrap_samples)

        _, _, test_original_accuracy_curve, test_bootstrap_accuracy_curve_list = generate_bootstrap_accuracy_curve(test_ema_predictions, test_labels, num_bootstrap_samples)
    else:
        raise NameError('Unsupported report type')
        
    
    
    #write the max validation criterion and the corresponding test criterion to file                      
    write_stats(valid_original_accuracy_curve, test_original_accuracy_curve, test_predictions_file_list, file_writer, predictions_dir, result_save_dir, report_type, task_name)
                      
    
    train_upper_percentile_curve, train_lower_percentile_curve = get_percentile_curve(train_bootstrap_accuracy_curve_list, bootstrap_upper_percentile, bootstrap_lower_percentile)
    
    valid_upper_percentile_curve, valid_lower_percentile_curve = get_percentile_curve(valid_bootstrap_accuracy_curve_list, bootstrap_upper_percentile, bootstrap_lower_percentile)
    
    test_upper_percentile_curve, test_lower_percentile_curve = get_percentile_curve(test_bootstrap_accuracy_curve_list, bootstrap_upper_percentile, bootstrap_lower_percentile)
    
  
    save_diagnosis_plots_multitask(figure_title, result_save_dir, test_original_accuracy_curve, test_lower_percentile_curve, test_upper_percentile_curve, valid_original_accuracy_curve, valid_lower_percentile_curve, valid_upper_percentile_curve, total_loss, unweighted_diagnosis_loss, weighted_diagnosis_loss, unweighted_view_loss, weighted_view_loss, scaled_weighted_view_loss, ylim_lower, ylim_upper, report_type)
    