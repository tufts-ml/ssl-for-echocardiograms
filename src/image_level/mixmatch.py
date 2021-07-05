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
"""MixMatch training.
- Ensure class consistency by producing a group of `nu` augmentations of the same image and guessing the label for the
  group.
- Sharpen the target distribution.
- Use the sharpened distribution directly as a smooth label in MixUp.
"""

import functools
import os
import numpy as np

import sys
from absl import app
from absl import flags
from easydict import EasyDict
from libml import layers, utils, models
import libml.data_pair as data_pair
from libml.layers import MixMode
from libml.result_analysis import perform_analysis
from libml.checkpoint_ensemble import perform_ensemble
import tensorflow as tf



FLAGS = flags.FLAGS


class MixMatch(models.MultiModel):

    def augment(self, x, l, beta, **kwargs):
        assert 0, 'Do not call.'

    def guess_label(self, y, classifier, T, **kwargs):
        del kwargs
        logits_y = [classifier(yi, training=True) for yi in y]
        logits_y = tf.concat(logits_y, 0)
        # Compute predicted probability distribution py.
        p_model_y = tf.reshape(tf.nn.softmax(logits_y), [len(y), -1, self.nclass])
        p_model_y = tf.reduce_mean(p_model_y, axis=0)
        # Compute the target distribution.
        p_target = tf.pow(p_model_y, 1. / T)
        p_target /= tf.reduce_sum(p_target, axis=1, keep_dims=True)
        return EasyDict(p_target=p_target, p_model=p_model_y)

    def model(self, batch, lr, wd, ema, class_weights, beta, w_match, warmup_kimg=1024, warmup_delay=0, nu=2, mixmode='xxy.yxy', **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x') #labeled images, after augmentation operation in data_pair.py
        y_in = tf.placeholder(tf.float32, [None, nu] + hwc, 'y') #unlabeled images, after augmentation operation in data_pair.py
        l_in = tf.placeholder(tf.int32, [None], 'labels') #label of the labeled images
        ul_in = tf.placeholder(tf.int32, [None, nu], 'unlabeled_labels') #label of the unlabeled images
        
        class_weights = class_weights.split(',')
        class_weights = [float(i) for i in class_weights]
        class_weights = tf.constant(class_weights)
        weights = tf.reduce_sum(class_weights * tf.one_hot(l_in, self.nclass))
        
        wd *= lr
        w_match *= tf.clip_by_value(tf.cast(self.step - (warmup_delay << 10), tf.float32) / (warmup_kimg << 10), 0, 1)
        augment = MixMode(mixmode)
        classifier = functools.partial(self.classifier, **kwargs)

        y = tf.reshape(tf.transpose(y_in, [1, 0, 2, 3, 4]), [-1] + hwc)
        guess = self.guess_label(tf.split(y, nu), classifier, T=0.5, **kwargs)
        ly = tf.stop_gradient(guess.p_target)
        lx = tf.one_hot(l_in, self.nclass)
        
        #perform mixup
        xy, labels_xy = augment([x_in] + tf.split(y, nu), [lx] + [ly] * nu, [beta, beta])
        x, y = xy[0], xy[1:]
        labels_x, labels_y = labels_xy[0], tf.concat(labels_xy[1:], 0)
        
        del xy
        
        batches = layers.interleave([x] + y, batch)
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        logits = [classifier(batches[0], training=True)]
        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
        for batchi in batches[1:]:
            logits.append(classifier(batchi, training=True))
        logits = layers.interleave(logits, batch)
        logits_x = logits[0]
        logits_y = tf.concat(logits[1:], 0)

        unweighted_loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_x, logits=logits_x)
        loss_xe = unweighted_loss_xe * weights
        
        loss_xe = tf.reduce_mean(loss_xe)
        loss_l2u = tf.square(labels_y - tf.nn.softmax(logits_y))
        loss_l2u = tf.reduce_mean(loss_l2u)
        

        tf.summary.scalar('losses/xe', loss_xe)
        tf.summary.scalar('losses/unweighted_xe', tf.py_func(np.mean, [unweighted_loss_xe], tf.float32))

        tf.summary.scalar('losses/l2u_plain', loss_l2u)
        tf.summary.scalar('losses/l2u_scaled', loss_l2u * w_match) #also track the unlabeled loss after multiplied by unlabeled loss coefficient

        #######################################################################################################################
        
        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        
        vars_to_train = self.get_variables_to_train()
        
        train_op = tf.train.AdamOptimizer(lr).minimize(loss_xe + w_match * loss_l2u, var_list=vars_to_train, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        # Tuning op: only retrain batch norm.
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        classifier(batches[0], training=True)
        train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                              if v not in skip_ops])

        return EasyDict(
            x=x_in, y=y_in, label=l_in, unlabeled_label=ul_in, train_op=train_op, tune_op=train_bn,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)),
            labeled_losses = loss_xe,
            unlabeled_losses_unscaled = loss_l2u,
            unlabeled_losses_multiplier = w_match,
            unlabeled_losses_scaled = loss_l2u * w_match)


def main(argv):
    del argv  # Unused.
    assert FLAGS.nu == 2
    
    
    ######################################################################################
    #experiment settings:
    nclass=3 #to manually define in the script
    height=64
    width=64
    colors=1
    figure_title = 'MixMatch'
    num_bootstrap_samples = 200 #how many bootstrap samples to use
    bootstrap_upper_percentile = 90 #what upper percentile of the bootstrap result to show
    bootstrap_lower_percentile = 10 #what lower percentile of the bootstrap result to show
    num_selection_step = 80 #number of forward stepwise selection step to perform
    ensemble_last_checkpoints = 25 #use last 25 checkpoints as source for ensemble
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
    DATASETS.update([data_pair.DataSet.creator('echo', train_labeled_files, train_unlabeled_files, valid_files, test_files, [data_pair.augment_echo, data_pair.stack_augment(data_pair.augment_echo)], nclass=nclass, height=height, width=width, colors=colors)])
    
    dataset = DATASETS[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = MixMatch(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        class_weights=FLAGS.class_weights,
        beta=FLAGS.beta,
        w_match=FLAGS.w_match,
        warmup_kimg=FLAGS.warmup_kimg,
        warmup_delay=FLAGS.warmup_delay,
        mixmode=FLAGS.mixmode,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    
    
    experiment_dir = model.train_dir
    
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)

    
#     for report_type in ['RAW_BalancedAccuracy', 'EMA_BalancedAccuracy']:
    result_save_dir = os.path.join(experiment_dir, 'result_analysis', FLAGS.report_type)

    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)

    perform_analysis(figure_title, experiment_dir, result_save_dir, num_bootstrap_samples, bootstrap_upper_percentile, bootstrap_lower_percentile, ylim_lower, ylim_upper, FLAGS.report_type, FLAGS.task_name)

    perform_ensemble(experiment_dir, result_save_dir, num_selection_step, FLAGS.report_type, ensemble_last_checkpoints)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_string('class_weights', '0.2031,0.7763,0.0205', 'the weights used for weighted cross entropy loss')
    flags.DEFINE_float('beta', 0.75, 'Mixup beta distribution.')
    flags.DEFINE_float('w_match', 100, 'Weight for distribution matching loss.')
    flags.DEFINE_integer('scales', 4, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_string('mixmode', 'xxy.yxy', 'using what mixing pattern for MixMatch')
    flags.DEFINE_integer('warmup_kimg', 1024, 'steps when consistency loss ramup schedule reach max')
    flags.DEFINE_integer('warmup_delay', 0, 'delay the warmup schedule for certain steps')
    flags.DEFINE_string('task_name', 'ViewClassification', 'either ViewClassification or DiagnosisClassification')
    flags.DEFINE_string('train_labeled_files', 'train-label_VIEW.tfrecord', 'name of the train labeled tfrecord')
    flags.DEFINE_string('valid_files', 'valid_VIEW.tfrecord', 'name of the valid tfrecord')
    flags.DEFINE_string('test_files', 'test_VIEW.tfrecord', 'name of the test tfrecord')   
    flags.DEFINE_string('train_unlabeled_files', 'train-unlabel_VIEW.tfrecord', 'name of the unlabeled set tfrecord')
    FLAGS.set_default('dataset', 'echo')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
    
    