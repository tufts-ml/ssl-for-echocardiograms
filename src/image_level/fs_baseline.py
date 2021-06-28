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

from absl import app
from absl import flags
from easydict import EasyDict

import libfs.data as datafs
from libfs.train import ClassifyFullySupervised
from libml import utils
from libml.models import MultiModel
from libml.result_analysis import perform_analysis
from libml.checkpoint_ensemble import perform_ensemble

import tensorflow as tf

FLAGS = flags.FLAGS


class FSBaseline(ClassifyFullySupervised, MultiModel):

    def model(self, lr, wd, ema, class_weights, **kwargs):
        
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x') 
        l_in = tf.placeholder(tf.int32, [None], 'labels') 
        wd *= lr
        l = tf.one_hot(l_in, self.nclass)
        
        class_weights = class_weights.split(',')
        class_weights = [float(i) for i in class_weights]
        class_weights = tf.constant(class_weights) #passed in class_weights is a list of floats
        weights = tf.reduce_sum(class_weights * l, axis = 1) #deduce weights fo batch samples based on their true label, l is a batch of one-hot labels
        
        x, l = self.augment(x_in, l, **kwargs)
        
        classifier = functools.partial(self.classifier, **kwargs)
        logits = classifier(x, training=True)
        
        unweighted_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l, logits=logits)
        loss = unweighted_loss * weights 
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('losses/xe', loss)
        tf.summary.scalar('losses/unweighted_xe', tf.py_func(np.mean, [unweighted_loss], tf.float32))
        

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

        return EasyDict(
            x=x_in, label=l_in, train_op=train_op, tune_op=train_bn,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)),
            labeled_losses = loss)


def main(argv):
    del argv  # Unused.
    
    ######################################################################################
    #experiment settings:
    nclass=3 #to manually define in the script
    height=64
    width=64
    colors=1
    figure_title = 'FS'
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
                                   datafs.data.augment_echo, nclass=nclass, height=height, width=width, colors=colors)])

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
        class_weights=FLAGS.class_weights,
        continued_training=FLAGS.continued_training,
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
    flags.DEFINE_string('class_weights', '0.2031,0.7763,0.0205', 'the weights used for weighted cross entropy loss')
    flags.DEFINE_string('continued_training', '0_30000', 'the job is which step to which step') 
    flags.DEFINE_string('task_name', 'ViewClassification', 'either ViewClassification or DiagnosisClassification')
    flags.DEFINE_string('train_labeled_files', 'train-label_VIEW.tfrecord', 'name of the train labeled tfrecord')
    flags.DEFINE_string('valid_files', 'valid_VIEW.tfrecord', 'name of the valid tfrecord')
    flags.DEFINE_string('test_files', 'test_VIEW.tfrecord', 'name of the test tfrecord')   
    flags.DEFINE_float('smoothing', 0.001, 'Label smoothing.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    FLAGS.set_default('dataset', 'echo')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
