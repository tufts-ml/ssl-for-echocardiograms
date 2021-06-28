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
"""Fully supervised training.
"""

import functools
import os
import numpy as np

import sys

from absl import app
from absl import flags
from easydict import EasyDict

import libfs.data as datafs
from libfs.train_multitask import ClassifyFullySupervised
from libml import utils
from libml.models import MultiModel
from libml.result_analysis import perform_analysis_multitask as perform_analysis
from libml.checkpoint_ensemble import perform_ensemble

import tensorflow as tf

FLAGS = flags.FLAGS


class FSBaseline(ClassifyFullySupervised, MultiModel):

    def model(self, lr, wd, ema, class_weights_diagnosis, class_weights_view, **kwargs):
        
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x') #x_in already applied standard data augmentation
        l_in_diagnosis = tf.placeholder(tf.int32, [None], 'diagnosis_labels')
        l_in_view = tf.placeholder(tf.int32, [None], 'view_labels')
        
        wd *= lr
        l_diagnosis = tf.one_hot(l_in_diagnosis, self.nclass)
        l_view = tf.one_hot(l_in_view, self.nclass)
        
        class_weights_diagnosis = class_weights_diagnosis.split(',')
        class_weights_diagnosis = [float(i) for i in class_weights_diagnosis]
        class_weights_diagnosis = tf.constant(class_weights_diagnosis) #passed in class_weights is a list of floats
        weights_diagnosis = tf.reduce_sum(class_weights_diagnosis * l_diagnosis, axis = 1) #deduce weights fo batch samples based on their true label, l is a batch of one-hot labels
        
        class_weights_view = class_weights_view.split(',')
        class_weights_view = [float(i) for i in class_weights_view]
        class_weights_view = tf.constant(class_weights_view) #passed in class_weights is a list of floats
        weights_view = tf.reduce_sum(class_weights_view * l_view, axis = 1) #deduce weights fo batch samples based on their true label, l is a batch of one-hot labels
        
        smoothing = kwargs['smoothing']
        l_diagnosis = l_diagnosis - smoothing * (l_diagnosis - 1./self.nclass)
        l_view = l_view - smoothing * (l_view - 1./self.nclass)
        
        classifier = functools.partial(self.classifier, **kwargs)
        diagnosis_logits, view_logits = classifier(x_in, training=True)
        
        #diagnosis loss:
        unweighted_diagnosis_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l_diagnosis, logits=diagnosis_logits)
        weighted_diagnosis_loss = unweighted_diagnosis_loss * weights_diagnosis 
        weighted_diagnosis_loss = tf.reduce_mean(weighted_diagnosis_loss)
        
        tf.summary.scalar('losses/weighted_diagnosis_loss', weighted_diagnosis_loss)
        tf.summary.scalar('losses/unweighted_diagnosis_loss', tf.py_func(np.mean, [unweighted_diagnosis_loss], tf.float32))
        
        #view loss:
        auxiliary_task_weight = kwargs['auxiliary_task_weight']
        unweighted_view_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l_view, logits=view_logits)
        weighted_view_loss = unweighted_view_loss * weights_view 
        weighted_view_loss = tf.reduce_mean(weighted_view_loss)
        scaled_weighted_view_loss = auxiliary_task_weight * weighted_view_loss
        
        tf.summary.scalar('losses/weighted_view_loss', weighted_view_loss)
        tf.summary.scalar('losses/scaled_weighted_view_loss', scaled_weighted_view_loss)
        tf.summary.scalar('losses/unweighted_view_loss', tf.py_func(np.mean, [unweighted_view_loss], tf.float32))
        
        #total loss:
        loss = weighted_diagnosis_loss + scaled_weighted_view_loss
        tf.summary.scalar('losses/total_loss', weighted_view_loss)

        #######################################################################################################################

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) + [ema_op]
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        train_op = tf.train.AdamOptimizer(lr).minimize(loss, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        # Tuning op: only retrain batch norm.
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        classifier(x_in, training=True)
        train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                              if v not in skip_ops])

        classify_raw_logits, _ = classifier(x_in, training=False)
        classify_op_logits, _ = classifier(x_in, getter=ema_getter, training=False)
        
        return EasyDict(
            x=x_in, label_diagnosis=l_in_diagnosis, label_view=l_in_view, train_op=train_op, tune_op=train_bn,
            classify_raw=tf.nn.softmax(classify_raw_logits),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classify_op_logits),
            total_loss=loss, unweighted_diagnosis_loss=unweighted_diagnosis_loss, weighted_diagnosis_loss=weighted_diagnosis_loss, unweighted_view_loss=unweighted_view_loss, weighted_view_loss=weighted_view_loss, scaled_weighted_view_loss=scaled_weighted_view_loss)


def main(argv):
    del argv  # Unused.
    
    ######################################################################################
    #experiment settings:
    nclass=3 #to manually define in the script
    height=64
    width=64
    colors=1
    figure_title = 'FS_multitask'
    num_bootstrap_samples = 200 #how many bootstrap samples to use
    bootstrap_upper_percentile = 90 #what upper percentile of the bootstrap result to show
    bootstrap_lower_percentile = 10 #what lower percentile of the bootstrap result to show
    num_selection_step = 80 #number of forward stepwise selection step to perform
    ensemble_last_checkpoints = 25 #use last 100 checkpoints as source for ensemble
    ylim_lower=40
    ylim_upper=100
    
    train_labeled_files = FLAGS.train_labeled_files.split(',')
    valid_files = FLAGS.valid_files.split(',')
    test_files = FLAGS.test_files.split(',')
    
#     print('train_labeled_files is {}'.format(train_labeled_files), flush=True)
#     print('valid_files is {}'.format(valid_files), flush=True)
#     print('test_files is {}'.format(test_files), flush=True)
    ######################################################################################

    
    DATASETS = {}
    DATASETS.update([datafs.DataSetFS.creator('echo', train_labeled_files, valid_files, test_files,
                                   datafs.data.augment_echo_multitask, parse_fn=datafs.data.default_parse_multitask, memoize_fn=datafs.data.memoize_multitask, nclass=nclass, height=height, width=width, colors=colors)])

    dataset = DATASETS[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = FSBaseline(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        class_weights_diagnosis=FLAGS.class_weights_diagnosis,
        class_weights_view=FLAGS.class_weights_view,
        continued_training=FLAGS.continued_training,
        auxiliary_task_weight=FLAGS.auxiliary_task_weight,
        smoothing=FLAGS.smoothing,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    
    experiment_dir = model.train_dir
    
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)
    
    
    for report_type in ['RAW_BalancedAccuracy', 'EMA_BalancedAccuracy']:
        result_save_dir = os.path.join(experiment_dir, 'result_analysis', report_type)

        if not os.path.exists(result_save_dir):
            os.makedirs(result_save_dir)

        perform_analysis(figure_title, experiment_dir, result_save_dir, num_bootstrap_samples, bootstrap_upper_percentile, bootstrap_lower_percentile, ylim_lower, ylim_upper, report_type, FLAGS.task_name)
        
        perform_ensemble(experiment_dir, result_save_dir, num_selection_step, report_type, ensemble_last_checkpoints)
        
        
    
if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_string('class_weights_diagnosis', '0.3385,0.3292,0.3323', 'the weights used for weighted cross entropy loss for diagnosis prediction')
    flags.DEFINE_string('class_weights_view', '0.2447,0.7238,0.0316', 'the weights used for weighted cross entropy loss for view prediction')
    flags.DEFINE_float('auxiliary_task_weight', 0.3, 'control the strength of auxiliary task loss')
    flags.DEFINE_string('continued_training', '0_30000', 'the job is which step to which step')
    flags.DEFINE_string('task_name', 'ViewClassification', 'either ViewClassification or DiagnosisClassification')
    flags.DEFINE_string('train_labeled_files', 'train-label_VIEW.tfrecord', 'name of the train labeled tfrecord')
    flags.DEFINE_string('valid_files', 'valid_VIEW.tfrecord', 'name of the valid tfrecord')
    flags.DEFINE_string('test_files', 'test_VIEW.tfrecord', 'name of the test tfrecord')
    flags.DEFINE_float('smoothing', 0.001, 'Label smoothing.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    FLAGS.set_default('arch', 'resnet_multitask')
    FLAGS.set_default('dataset', 'echo')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
