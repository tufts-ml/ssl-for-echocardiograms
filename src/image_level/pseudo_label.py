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
"""Pseudo-label: The simple and efficient semi-supervised learning method fordeep neural networks.

Reimplementation of http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf
"""

import functools
import os

import sys
import numpy as np
from absl import app
from absl import flags
from easydict import EasyDict
from libml import utils, data, models
from libml.result_analysis import perform_analysis
from libml.checkpoint_ensemble import perform_ensemble
import tensorflow as tf

FLAGS = flags.FLAGS


class PseudoLabel(models.MultiModel):

    def model(self, lr, wd, ema, class_weights, warmup_pos, warmup_delay, consistency_weight, threshold, **kwargs):

        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x') #labeled images, after augmentation operation in data_pair.py
        y_in = tf.placeholder(tf.float32, [None] + hwc, 'y') #unlabeled images, after augmentation operation in data_pair.py
        l_in = tf.placeholder(tf.int32, [None], 'labels') #label of the labeled images
        ul_in = tf.placeholder(tf.int32, [None], 'unlabeled_labels') #label of the unlabeled images
        
        l = tf.one_hot(l_in, self.nclass)
        
        class_weights = class_weights.split(',')
        class_weights = [float(i) for i in class_weights]
        class_weights = tf.constant(class_weights)
        weights = tf.reduce_sum(class_weights * l)        
        
        wd *= lr
        warmup = tf.clip_by_value(tf.cast(self.step - (warmup_delay << 10), tf.float32) / (warmup_pos * (FLAGS.train_kimg << 10)), 0, 1)

        classifier = functools.partial(self.classifier, **kwargs)
        logits_x = classifier(x_in, training=True)
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Take only first call to update batch norm.
        logits_y = classifier(y_in, training=True)
        # Get the pseudo-label loss
        loss_pl = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(logits_y, axis=-1), logits=logits_y
        )
        # Masks denoting which data points have high-confidence predictions
        greater_than_thresh = tf.reduce_any(
            tf.greater(tf.nn.softmax(logits_y), threshold),
            axis=-1,
            keepdims=True,
        )
        greater_than_thresh = tf.cast(greater_than_thresh, loss_pl.dtype)
        # Only enforce the loss when the model is confident
        loss_pl *= greater_than_thresh
        # Note that we also average over examples without confident outputs;
        # this is consistent with the realistic evaluation codebase
        loss_pl = tf.reduce_mean(loss_pl)

        unweighted_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l, logits=logits_x)
        loss = unweighted_loss * weights
        loss = tf.reduce_mean(loss)
        
        tf.summary.scalar('losses/xe', loss)
        tf.summary.scalar('losses/unweighted_xe', tf.py_func(np.mean, [unweighted_loss], tf.float32))

        tf.summary.scalar('losses/pl_plain', loss_pl)
        tf.summary.scalar('losses/pl_scaled', loss_pl * warmup * consistency_weight) #also track the unlabeled loss after multiplied by unlabeled loss coefficient

        
        #######################################################################################################################
        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        train_op = tf.train.AdamOptimizer(lr).minimize(loss + loss_pl * warmup * consistency_weight,
                                                       colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        # Tuning op: only retrain batch norm.
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        classifier(x_in, training=True)
        train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                              if v not in skip_ops])

        return EasyDict(
            x=x_in, y=y_in, label=l_in, unlabeled_label=ul_in, train_op=train_op, tune_op=train_bn,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)),
            labeled_losses = loss,
            unlabeled_losses_unscaled = loss_pl,
            unlabeled_losses_multiplier = warmup * consistency_weight,
            unlabeled_losses_scaled = loss_pl * warmup * consistency_weight)


def main(argv):
    del argv  # Unused.
    
    
    ######################################################################################
    #experiment settings:
    nclass=3 #to manually define in the script
    height=64
    width=64
    colors=1
    figure_title = 'PL'
    num_bootstrap_samples = 10 #how many bootstrap samples to use
    bootstrap_upper_percentile = 90 #what upper percentile of the bootstrap result to show
    bootstrap_lower_percentile = 10 #what lower percentile of the bootstrap result to show
    num_selection_step = 80 #number of forward stepwise selection step to perform
    ensemble_last_checkpoints = 25 #use last 100 checkpoints as source for ensemble
    ylim_lower=40
    ylim_upper=100
    
    train_labeled_files = FLAGS.train_labeled_files.split(',')
    train_unlabeled_files = FLAGS.train_unlabeled_files.split(',')
    valid_files = FLAGS.valid_files.split(',')
    test_files = FLAGS.test_files.split(',')
    
#     print('train_labeled_files is {}'.format(train_labeled_files), flush=True)
#     print('train_unlabeled_files is {}'.format(train_unlabeled_files), flush=True)
#     print('valid_files is {}'.format(valid_files), flush=True)
#     print('test_files is {}'.format(test_files), flush=True)
    
    
    ######################################################################################
        
    DATASETS = {}
    DATASETS.update([data.DataSet.creator('echo', train_labeled_files, train_unlabeled_files, valid_files, test_files, data.augment_echo, nclass=nclass, height=height, width=width, colors=colors)])
    
    dataset = DATASETS[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = PseudoLabel(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        warmup_pos=FLAGS.warmup_pos,
        warmup_delay=FLAGS.warmup_delay,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        class_weights=FLAGS.class_weights,
        smoothing=FLAGS.smoothing,
        consistency_weight=FLAGS.consistency_weight,
        threshold=FLAGS.threshold,
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
    flags.DEFINE_float('consistency_weight', 1., 'Consistency weight.')
    flags.DEFINE_float('threshold', 0.95, 'Pseudo-label threshold.')
    flags.DEFINE_float('warmup_pos', 0.4, 'Relative position at which constraint loss warmup ends.')
    flags.DEFINE_integer('warmup_delay', 0, 'delay the warmup schedule for certain steps')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_string('class_weights', '0.2031,0.7763,0.0205', 'the weights used for weighted cross entropy loss')
    flags.DEFINE_float('smoothing', 0.1, 'Label smoothing.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_string('task_name', 'ViewClassification', 'either ViewClassification or DiagnosisClassification')
    flags.DEFINE_string('train_labeled_files', 'train-label_VIEW.tfrecord', 'name of the train labeled tfrecord')
    flags.DEFINE_string('valid_files', 'valid_VIEW.tfrecord', 'name of the valid tfrecord')
    flags.DEFINE_string('test_files', 'test_VIEW.tfrecord', 'name of the test tfrecord')   
    flags.DEFINE_string('train_unlabeled_files', 'train-unlabel_VIEW.tfrecord', 'name of the unlabeled set tfrecord')
    FLAGS.set_default('dataset', 'cifar10.3@250-5000')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
