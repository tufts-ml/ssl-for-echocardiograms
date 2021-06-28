import argparse
import os
import sys
from libml.utils import retrieve_TestAccuracy_at_MaxValidationAccuracy, simple_bagging, init_composition, ensemble, test_prediction_forward_stepwise, save_pickle



def perform_ensemble(experiment_dir, result_save_dir, num_selection_step, report_type, ensemble_last_checkpoints=100):
    
    #define predictions_dir
    predictions_dir = os.path.join(experiment_dir, 'predictions')
    
    print('currently pointing to predictions_dir: {}'.format(predictions_dir), flush=True)
    print('currently saving ensemble results to: {}'.format(result_save_dir), flush=True)
    print('performing {} selection step'.format(num_selection_step), flush = True)
    
    
    valid_predictions_file_list = sorted([file for file in os.listdir(predictions_dir) if file.startswith('valid')], key = lambda s: int(s.split('_')[2]))
    
    test_predictions_file_list = sorted([file for file in os.listdir(predictions_dir) if file.startswith('test')], key = lambda s: int(s.split('_')[2]))
    
    assert len(valid_predictions_file_list) == len(test_predictions_file_list), 'number test pkl not equal to number valid pkl'

    
    #################################################without ensemble#################################################
    max_valid_accuracy, max_valid_accuracy_epoch, test_accuracy_at_max_valid_epoch = retrieve_TestAccuracy_at_MaxValidationAccuracy(predictions_dir, valid_predictions_file_list, test_predictions_file_list, report_type)
    
    
    #######################################################ensemble#######################################################
    valid_predictions_used_for_ensemble = valid_predictions_file_list[-ensemble_last_checkpoints:]
    test_predictions_used_for_ensemble = test_predictions_file_list[-ensemble_last_checkpoints:]

    ####################################################simple bagging####################################################
    bagging_valid_accuracy, _ = simple_bagging(valid_predictions_used_for_ensemble, predictions_dir, report_type)
    bagging_test_accuracy, _ = simple_bagging(test_predictions_used_for_ensemble, predictions_dir, report_type)
    
    #################################forward stepwise ensemble selection on validation set#################################
    initial_composition = init_composition(valid_predictions_used_for_ensemble)
    
    valid_composition, valid_best_ensemble_accuracy, _ = ensemble(predictions_dir, result_save_dir, num_selection_step, valid_predictions_used_for_ensemble, initial_composition, report_type)
    
    test_ensemble_accuracy, test_ensemble_predictions, test_composition = test_prediction_forward_stepwise(predictions_dir, valid_composition['best_composition'], report_type)
    
  
    
    #print results:
    header = r'''
    ################################################################
    Result for: {}
    '''
    experiment_name = predictions_dir.split('/')[1] + '_' + predictions_dir.split('/')[2] + '_' + predictions_dir.split('/')[3] + '_' + predictions_dir.split('/')[4]
    print(header.format(experiment_name))
    
    if report_type == 'RAW_BalancedAccuracy':
        output_string1 = 'NO ensemble: max RAW validation balanced accuracy is {}, RAW test balanced accuracy at max RAW validation balanced accuracy is {}'
        output_string2 = 'Bagging, last {} checkpoints: RAW valid balanced accuracy is {}, RAW test balanced accuracy is {}'
        output_string3 = 'Forward Stepwise, last {} checkpoints: RAW valid balanced accuracy is {}, RAW test balanced accuracy is {}'
        
    elif report_type == 'EMA_BalancedAccuracy':
        output_string1 = 'NO ensemble: max EMA validation balanced accuracy is {}, EMA test balanced accuracy at max EMA validation balanced accuracy is {}'
        output_string2 = 'Bagging, last {} checkpoints: EMA valid balanced accuracy is {}, EMA test balanced accuracy is {}'
        output_string3 = 'Forward Stepwise, last {} checkpoints: EMA valid balanced accuracy is {}, EMA test balanced accuracy is {}'
    
    else:
        raise NameError('Unsupported report type')


    #write to file
    file_writer = open(os.path.join(result_save_dir, 'Ensemble_performance.txt'), 'w')
    file_writer.write(report_type)
    file_writer.write(output_string1.format(max_valid_accuracy, test_accuracy_at_max_valid_epoch ))
    file_writer.write(output_string2.format(ensemble_last_checkpoints, bagging_valid_accuracy, bagging_test_accuracy))
    file_writer.write(output_string3.format(ensemble_last_checkpoints, valid_best_ensemble_accuracy, test_ensemble_accuracy))
    
    #save the ensemble predictions
    save_pickle(result_save_dir, 'test_Ensemble_predictions.pkl', test_ensemble_predictions)
    save_pickle(result_save_dir, 'val_Ensemble_predictions.pkl', valid_composition['best_composition_prediction'])
    
    
