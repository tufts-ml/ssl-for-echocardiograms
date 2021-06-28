# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities."""

import glob
import os
import re

import tensorflow as tf
from tensorflow.python.client import device_lib
from absl import flags
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_cm
import pickle
import matplotlib.pyplot as plt

_GPUS = None
FLAGS = flags.FLAGS
flags.DEFINE_bool('log_device_placement', False, 'For debugging purpose.')


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
    
    
def calculate_accuracy(true_labels, predicted_labels):
    
    accuracy = (predicted_labels == true_labels).mean()*100
    
    return accuracy


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
        raise NameError('Unsupported return_type in hz_utils calculate_balanced_accuracy fn')

    
def get_config():
    config = tf.ConfigProto()
    if len(get_available_gpus()) > 1:
        config.allow_soft_placement = True
    if FLAGS.log_device_placement:
        config.log_device_placement = True
    config.gpu_options.allow_growth = True
    return config


def setup_tf():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.ERROR)


def smart_shape(x):
    s = x.shape
    st = tf.shape(x)
    return [s[i] if s[i].value is not None else st[i] for i in range(4)]


def ilog2(x):
    """Integer log2."""
    return int(np.ceil(np.log2(x)))


def find_latest_checkpoint(dir, glob_term='model.ckpt-*.meta'):
    """Replacement for tf.train.latest_checkpoint.

    It does not rely on the "checkpoint" file which sometimes contains
    absolute path and is generally hard to work with when sharing files
    between users / computers.
    """
    
    print('!!!!!!!!!!!Inside utils.py, find_latest_checkpoint is called, passed in dir is {}!!!!!!!!!!!'.format(dir), flush = True)

    r_step = re.compile('.*model\.ckpt-(?P<step>\d+)\.meta')
    matches = glob.glob(os.path.join(dir, glob_term))
    print('!!!!!!!!!!!Inside utils.py find_latest_checkpoint, matches is {}!!!!!!!!!!!'.format(matches), flush = True)
    
    matches = [(int(r_step.match(x).group('step')), x) for x in matches]
    print('!!!!!!!!!!!Inside utils.py find_latest_checkpoint, matches is {}!!!!!!!!!!!'.format(matches), flush = True)

    ckpt_file = max(matches)[1][:-5]
    print('!!!!!!!!!!!Inside utils.py find_latest_checkpoint, final returned ckpt_file is {}!!!!!!!!!!!'.format(ckpt_file), flush = True)

    return ckpt_file


def get_latest_global_step(dir):
    """Loads the global step from the latest checkpoint in directory.
  
    Args:
      dir: string, path to the checkpoint directory.
  
    Returns:
      int, the global step of the latest checkpoint or 0 if none was found.
    """
    
    print('!!!!!!!!!!!Inside utils.py, get_latest_global_step is called, passed in dir is {}!!!!!!!!!!!'.format(dir), flush = True)
    try:
        print('!!!!!!!!!!!Inside utils get_latest_global_step: try executed!!!!!!!!!!!', flush = True)
        checkpoint_reader = tf.train.NewCheckpointReader(find_latest_checkpoint(dir))
        
        print('!!!!!!!!!!!Inside utils get_latest_global_step, checkpoint_reader,get_tensor(tf.GraphKeys.GLOBAL_STEP) is {}!!!!!!!!!!!'.format(checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)), flush = True)
        return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
    except:  # pylint: disable=bare-except
        print('!!!!!!!!!!!Inside utils get_latest_global_step: except executed!!!!!!!!!!!', flush = True)
        return 0


def get_latest_global_step_in_subdir(dir):
    """Loads the global step from the latest checkpoint in sub-directories.

    Args:
      dir: string, parent of the checkpoint directories.

    Returns:
      int, the global step of the latest checkpoint or 0 if none was found.
    """
    print('!!!!!!!!!!!Inside utils.py, get_latest_global_step_IN_SUBDIR is called, passed in dir is {}!!!!!!!!!!!'.format(dir), flush = True)

    sub_dirs = (x for x in glob.glob(os.path.join(dir, '*')) if os.path.isdir(x))
    print('!!!!!!!!!!!Inside utils.py get_latest_global_step_IN_SUBDIR , sub_dirs is {}!!!!!!!!!!!'.format(sub_dirs), flush = True)

    step = 0
    for x in sub_dirs:
        step = max(step, get_latest_global_step(x))
    
    print('!!!!!!!!!!!Inside utils.py get_latest_global_step_IN_SUBDIR , final returned step is {}!!!!!!!!!!!'.format(step), flush = True)
    return step


def getter_ema(ema, getter, name, *args, **kwargs):
    """Exponential moving average getter for variable scopes.

    Args:
        ema: ExponentialMovingAverage object, where to get variable moving averages.
        getter: default variable scope getter.
        name: variable name.
        *args: extra args passed to default getter.
        **kwargs: extra args passed to default getter.

    Returns:
        If found the moving average variable, otherwise the default variable.
    """
    var = getter(name, *args, **kwargs)
    ema_var = ema.average(var)
    return ema_var if ema_var else var


def model_vars(scope=None):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


def gpu(x):
    return '/gpu:%d' % (x % max(1, len(get_available_gpus())))


def get_available_gpus():
    global _GPUS
    if _GPUS is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        local_device_protos = device_lib.list_local_devices(session_config=config)
        _GPUS = tuple([x.name for x in local_device_protos if x.device_type == 'GPU'])
    return _GPUS


def average_gradients(tower_grads):
    # Adapted from:
    #  https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. For each tower, a list of its gradients.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    if len(tower_grads) <= 1:
        return tower_grads[0]

    average_grads = []
    for grads_and_vars in zip(*tower_grads):
        grad = tf.reduce_mean([gv[0] for gv in grads_and_vars], 0)
        average_grads.append((grad, grads_and_vars[0][1]))
    return average_grads


def para_list(fn, *args):
    """Run on multiple GPUs in parallel and return list of results."""
    gpus = len(get_available_gpus())
    if gpus <= 1:
        return zip(*[fn(*args)])
    splitted = [tf.split(x, gpus) for x in args]
    outputs = []
    for gpu, x in enumerate(zip(*splitted)):
        with tf.name_scope('tower%d' % gpu):
            with tf.device(tf.train.replica_device_setter(
                    worker_device='/gpu:%d' % gpu, ps_device='/cpu:0', ps_tasks=1)):
                outputs.append(fn(*x))
    return zip(*outputs)


def para_mean(fn, *args):
    """Run on multiple GPUs in parallel and return means."""
    gpus = len(get_available_gpus())
    if gpus <= 1:
        return fn(*args)
    splitted = [tf.split(x, gpus) for x in args]
    outputs = []
    for gpu, x in enumerate(zip(*splitted)):
        with tf.name_scope('tower%d' % gpu):
            with tf.device(tf.train.replica_device_setter(
                    worker_device='/gpu:%d' % gpu, ps_device='/cpu:0', ps_tasks=1)):
                outputs.append(fn(*x))
    if isinstance(outputs[0], (tuple, list)):
        return [tf.reduce_mean(x, 0) for x in zip(*outputs)]
    return tf.reduce_mean(outputs, 0)


def para_cat(fn, *args):
    """Run on multiple GPUs in parallel and return concatenated outputs."""
    gpus = len(get_available_gpus())
    if gpus <= 1:
        return fn(*args)
    splitted = [tf.split(x, gpus) for x in args]
    outputs = []
    for gpu, x in enumerate(zip(*splitted)):
        with tf.name_scope('tower%d' % gpu):
            with tf.device(tf.train.replica_device_setter(
                    worker_device='/gpu:%d' % gpu, ps_device='/cpu:0', ps_tasks=1)):
                outputs.append(fn(*x))
    if isinstance(outputs[0], (tuple, list)):
        return [tf.concat(x, axis=0) for x in zip(*outputs)]
    return tf.concat(outputs, axis=0)


###################################################For performing ensemble############################################
def save_EnsembelAccuracy_VS_SelectionStep_plot(result_save_dir, num_steps, ensemble_accuracies, report_type):
    
        
    if report_type == 'RAW_BalancedAccuracy':
        ylabel_name = 'Ensemble RAW validation balanced accuracy'
        figure_title = 'Ensemble RAW validation balanced accuracy VS Number selection steps'
        
    
    elif report_type == 'EMA_BalancedAccuracy':
        ylabel_name = 'Ensemble EMA validation balanced accuracy'
        figure_title = 'Ensemble EMA validation balanced accuracy VS Number selection steps'
    
    else:
        raise NameError('Unsupported report type')
    
    plt.plot(list(range(1, num_steps + 1)), ensemble_accuracies)
    plt.ylabel(ylabel_name)
    plt.xlabel('Number selection steps')
    plt.title(figure_title)
    
    plt.savefig(os.path.join(result_save_dir, 'Validation_EnsemblePerformance_VS_SelectionStep.png'))
    #plt.show()
    plt.close()
    

def retrieve_TestAccuracy_at_MaxValidationAccuracy(predictions_dir, valid_predictions_file_list, test_predictions_file_list, report_type):
    
    '''
    Inside this function:
    
    "valid_accuracies", "test_accuracies", "max_valid_accuracy", "max_valid_accuracy_epoch", "test_accuracy_at_max_valid_epoch" generally refer to one of the RAW_Accuracy, RAW_BalancedAccuracy, EMA_Accuracy, EMA_BalancedAccuracy case.
    '''
    
    valid_accuracies = []
    test_accuracies = []
    
    
    if report_type == 'RAW_BalancedAccuracy':
        recorded_valid_accuracy_name = 'raw_balanced_accuracy'
        recorded_test_accuracy_name = 'raw_balanced_accuracy'
    
    elif report_type == 'EMA_BalancedAccuracy':
        recorded_valid_accuracy_name = 'ema_balanced_accuracy'
        recorded_test_accuracy_name = 'ema_balanced_accuracy'
    
    else:
        raise NameError('Unsupported report type (retrieve_TestAccuracy_at_MaxValidationAccuracy)')
        
    
    for valid_predictions_file in valid_predictions_file_list:
        valid_prediction_dict = load_pickle(predictions_dir, valid_predictions_file)
        valid_accuracies.append(valid_prediction_dict[recorded_valid_accuracy_name])
    valid_accuracies = np.array(valid_accuracies)
        
    for test_predictions_file in test_predictions_file_list:
        test_prediction_dict = load_pickle(predictions_dir, test_predictions_file)
        test_accuracies.append(test_prediction_dict[recorded_test_accuracy_name])
    test_accuracies = np.array(test_accuracies)
    
    max_valid_accuracy = np.max(valid_accuracies)
    max_valid_accuracy_epoch = np.argmax(valid_accuracies)
    test_accuracy_at_max_valid_epoch = test_accuracies[max_valid_accuracy_epoch]
    
    return max_valid_accuracy, max_valid_accuracy_epoch, test_accuracy_at_max_valid_epoch
    
    
def init_composition(model_library):
    '''
    model_library: ['valid_step_0_predictions.pkl',
                    'valid_step_65536_predictions.pkl',
                    'valid_step_131072_predictions.pkl',
                    'valid_step_196608_predictions.pkl',
                    'valid_step_262144_predictions.pkl',...,]
    '''
    ensemble_composition = {}
    for model_name in model_library:
        ensemble_composition[model_name] = 0
        
    return ensemble_composition


def forward_stepwise_selection(model_library, predictions_dir, running_ensemble_prediction, running_ensemble_size, report_type):
    '''
    peform one step selection: what model from the model library to add into the composition in this step
    '''
    
   
    
    if report_type == 'RAW_BalancedAccuracy':
        predictions_name = 'raw_predictions'
        accuracy_calculation_method = calculate_balanced_accuracy
        output_string = 'At current step, if adding {}, RAW validation balanced accuracy will be {}'
    
   
    elif report_type == 'EMA_BalancedAccuracy':
        predictions_name = 'ema_predictions'
        accuracy_calculation_method = calculate_balanced_accuracy
        output_string = 'At current step, if adding {}, EMA validation balanced accuracy will be {}'
    
    else:
        raise NameError('Unsupported report type (forward_stepwise_selection)')
        
    
    model_list = []
    ifadded_performance_list = []
    ifadded_ensemble_prediction_list = []
    
    for model_name in model_library:
        prediction_dict = load_pickle(predictions_dir, model_name)
        
        current_predictions = prediction_dict[predictions_name]
        current_true_labels = prediction_dict['true_labels']
        
        model_list.append(model_name)
        
        ensembled_prediction = (running_ensemble_prediction * running_ensemble_size + current_predictions) / (running_ensemble_size + 1)
        
        ifadded_ensemble_prediction_list.append(ensembled_prediction)
        ifadded_performance = accuracy_calculation_method(current_true_labels, ensembled_prediction.argmax(1))
        
        print(output_string.format(model_name, ifadded_performance))
        ifadded_performance_list.append(ifadded_performance)
        
    model_to_add_index = np.argmax(ifadded_performance_list)
    
    model_to_add = model_list[model_to_add_index]
    updated_running_ensemble_prediction = ifadded_ensemble_prediction_list[model_to_add_index]
    ensemble_accuracy = ifadded_performance_list[model_to_add_index]
    
    return model_to_add, updated_running_ensemble_prediction, ensemble_accuracy


def ensemble(predictions_dir, result_save_dir, num_steps, model_library, initial_composition, report_type):
    
    current_ensemble_composition = initial_composition
    current_ensemble_size = 0
    current_ensemble_prediction = 0
    ensemble_accuracies = []
    
    best_ensemble_accuracy = 0
    best_composition = {}
    
    for i in range(1, num_steps + 1):
        print('#########################ensemble step {}#########################\n'.format(i))
        
        model_to_add, updated_running_ensemble_prediction, ensemble_accuracy = forward_stepwise_selection(model_library, predictions_dir, current_ensemble_prediction, current_ensemble_size, report_type)
        current_ensemble_composition[model_to_add] += 1
        current_ensemble_size += 1
        
        current_ensemble_prediction = updated_running_ensemble_prediction
        ensemble_accuracies.append(ensemble_accuracy)
        
        if ensemble_accuracy > best_ensemble_accuracy:
            best_ensemble_accuracy = ensemble_accuracy
            best_composition['step'] = i
            best_composition['best_composition'] = current_ensemble_composition.copy()
            best_composition['best_composition_prediction'] = current_ensemble_prediction.copy()
        
        print('\n\n')
        
    save_EnsembelAccuracy_VS_SelectionStep_plot(result_save_dir, num_steps, ensemble_accuracies, report_type)
    
    return best_composition, best_ensemble_accuracy, ensemble_accuracies


def test_prediction_forward_stepwise(predictions_dir, best_composition_from_validation, report_type):
    '''
    take in the selected ensemble composition from validation, and ensemble test predictions accordingly
    '''
    
    
    if report_type == 'RAW_BalancedAccuracy':
        predictions_name = 'raw_predictions'
        accuracy_calculation_method = calculate_balanced_accuracy
    
    
    elif report_type == 'EMA_BalancedAccuracy':
        predictions_name = 'ema_predictions'
        accuracy_calculation_method = calculate_balanced_accuracy
    
    else:
        raise NameError('Unsupported report type (forward_stepwise_selection)')
                
    test_composition = {} # each element is the step number of the prediciton file
    
    total_model_used = 0
    sum_predictions = 0
    
    for valid_model_name, number_count in best_composition_from_validation.items():
        test_model_name = valid_model_name.replace('valid', 'test')
        test_composition[test_model_name] = number_count
        
        test_prediction_dict = load_pickle(predictions_dir, test_model_name)
        sum_predictions += test_prediction_dict[predictions_name] * number_count
        total_model_used += number_count
    
    true_labels = test_prediction_dict['true_labels']
    ensemble_predictions = sum_predictions/total_model_used
    
    test_ensemble_accuracy = accuracy_calculation_method(true_labels, ensemble_predictions.argmax(1))
    
    return test_ensemble_accuracy, ensemble_predictions, test_composition


#simple bagging can be directly used on test predictions
def simple_bagging(model_library, predictions_dir, report_type):
    
        
    if report_type == 'RAW_BalancedAccuracy':
        predictions_name = 'raw_predictions'
        accuracy_calculation_method = calculate_balanced_accuracy
        
    elif report_type == 'EMA_BalancedAccuracy':
        predictions_name = 'ema_predictions'
        accuracy_calculation_method = calculate_balanced_accuracy

    else:
        raise NameError('Unsupported report type')
    
    num_models = 0
    running_predictions = 0
    
    #sanity check: dimension
    prediction_array_dimension = None
    
    for model_name in model_library:
        num_models += 1
        
        prediction_dict = load_pickle(predictions_dir, model_name)
        
        if prediction_array_dimension is None:
            prediction_array_dimension = prediction_dict[predictions_name].shape
            assert prediction_dict['raw_predictions'].shape == prediction_dict['ema_predictions'].shape, 'raw_predictions and ema_predictions dimension mismatch'

        else:
            assert prediction_array_dimension == prediction_dict['raw_predictions'].shape == prediction_dict['ema_predictions'].shape, 'dimension mismatch'
        
        current_predictions = prediction_dict[predictions_name]
        
        running_predictions += current_predictions
    
    true_labels = prediction_dict['true_labels']
    
    average_predictions = running_predictions/num_models
    
    assert average_predictions.shape == running_predictions.shape == prediction_array_dimension, 'average_predictions, running_predictions and prediction_array_dimension mismatch'

    ensemble_accuracy = accuracy_calculation_method(true_labels, average_predictions.argmax(1))
    
    return ensemble_accuracy, average_predictions
