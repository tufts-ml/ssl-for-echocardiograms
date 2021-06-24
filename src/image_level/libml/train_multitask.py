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
"""Training loop, checkpoint saving and loading, evaluation code."""

import json
#import os.path
import os
import shutil

import numpy as np
import tensorflow as tf
from absl import flags
from easydict import EasyDict
from tqdm import trange

from libml_SuggestedMultitaskAblation import data, utils
from libml_SuggestedMultitaskAblation import hz_utils

FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', './experiments',
                    'Folder where to save training data.')
flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
flags.DEFINE_integer('batch', 64, 'Batch size.')
flags.DEFINE_integer('train_kimg', 1 << 14, 'Training duration in kibi-samples.')
flags.DEFINE_integer('report_kimg', 64, 'Report summary period in kibi-samples.')
flags.DEFINE_integer('save_kimg', 64, 'Save checkpoint period in kibi-samples.')
flags.DEFINE_integer('keep_ckpt', 50, 'Number of checkpoints to keep.')
flags.DEFINE_string('eval_ckpt', '', 'Checkpoint to evaluate. If provided, do not do training, just do eval.')
flags.DEFINE_bool('reset_global_step', False, 'initialized from pretrained weights')
flags.DEFINE_string('load_ckpt', "None", 'Checkpoint to initialize from')
flags.DEFINE_string('checkpoint_exclude_scopes', "None", 'Comma-separated list of scopes of variables to exclude when restoring')
flags.DEFINE_string('trainable_scopes', "None", 'Comma-separated list of scopes of variables to train')


class Model:
    def __init__(self, train_dir: str, dataset: data.DataSet, **kwargs):
        self.train_dir = os.path.join(train_dir, self.experiment_name(**kwargs))
        self.params = EasyDict(kwargs)
        self.dataset = dataset
        self.session = None
        self.tmp = EasyDict(print_queue=[], cache=EasyDict())
        self.step = tf.train.get_or_create_global_step()
        self.ops = self.model(**kwargs)
        self.ops.update_step = tf.assign_add(self.step, FLAGS.batch)
        self.add_summaries(**kwargs)

        self.batch_labels_dict = {'label_diagnosis':[], 'label_view':[], 'unlabeled':[]}
        self.losses_dict = {'total_loss':[], 'unweighted_diagnosis_loss':[], 'weighted_diagnosis_loss':[], 'unweighted_view_loss':[], 'weighted_view_loss':[], 'scaled_weighted_view_loss':[]}
        self.best_balanced_validation_accuracy_raw = 0 #initialize to 0
        self.best_balanced_validation_accuracy_ema = 0 #initialize to 0
        self.saver = tf.train.Saver()
        self.init_op = tf.global_variables_initializer()

        #Feb24:
        #if there is already checkpoint in the Model/tf folder, then meaning we are continue training, set FLAGS.load_ckpt to None, FLAGS.reset_global_step to False
        try:
            continue_training_ckpt = utils.find_latest_checkpoint(self.checkpoint_dir)
            print('Inside Model, continue_training_ckpt is {}'.format(continue_training_ckpt).center(100, '#'), flush = True)
        except:
            continue_training_ckpt = None
        
        if continue_training_ckpt is not None:
            print('Inside Model, overwriting FLAGS.load_ckpt and FLAGS.reset_global_step'.center(100, '#'), flush = True)
            FLAGS.load_ckpt = "None"
            FLAGS.reset_global_step = False
            print('FLAGS.load_ckpt is {}, FLAGS.reset_global_step is {}'.format(FLAGS.load_ckpt, FLAGS.reset_global_step), flush = True)
            
        
        
        if FLAGS.load_ckpt != "None":
            print("Finetuning from checkpoint: {}".format(FLAGS.load_ckpt), flush = True)

            vars_to_exclude = []
            scopes_to_exclude = []
            if FLAGS.reset_global_step:
                scopes_to_exclude.append('global_step:0')

            if FLAGS.checkpoint_exclude_scopes != "None":
                scopes_to_exclude.extend([scope.strip() for scope in FLAGS.checkpoint_exclude_scopes.split(',')])
                
            for v in tf.all_variables():
                for scope in scopes_to_exclude:
                    if scope in v.name.split('/'):
                        vars_to_exclude.append(v)

            
            print('Inside Model, scopes_to_exclude is {}\n\n'.format(scopes_to_exclude), flush = True)
            print('Inside Model, vars_to_exclude is {}\n\n'.format(vars_to_exclude), flush = True)
              
            vars_to_load = [v for v in tf.all_variables() if v not in vars_to_exclude]
#             vars_to_load = [v for v in tf.all_variables()]
            
            print('Inside Model, vars_to_load is {}\n\n'.format(vars_to_load), flush = True)
            self.finetuning_saver = tf.train.Saver(var_list=vars_to_load)
        
        
        if self.dataset.train_unlabeled is None:
            self.is_fs = True
        else:
            self.is_fs = False
        
        print(' Config '.center(80, '-'))
        print('train_dir', self.train_dir)
        print('%-32s %s' % ('Model', self.__class__.__name__))
        print('%-32s %s' % ('Dataset', dataset.name))
        for k, v in sorted(kwargs.items()):
            print('%-32s %s' % (k, v))
        print(' Model '.center(80, '-'))
        to_print = [tuple(['%s' % x for x in (v.name, np.prod(v.shape), v.shape)]) for v in utils.model_vars(None)]
        to_print.append(('Total', str(sum(int(x[1]) for x in to_print)), ''))
        sizes = [max([len(x[i]) for x in to_print]) for i in range(3)]
        fmt = '%%-%ds  %%%ds  %%%ds' % tuple(sizes)
        for x in to_print[:-1]:
            print(fmt % x)
        print()
        print(fmt % to_print[-1])
        print('-' * 80)
        self._create_initial_files()
        
        
    #hz added Feb20
    #ref: https://github.com/tensorflow/models/blob/9ff763923a2f0795c93c09a843f1ed5da4d556aa/research/slim/train_image_classifier.py#L394
    def get_variables_to_train(self):
        '''
        Return a list of variables to train, to be passed to optimizer
        '''
        
        if FLAGS.trainable_scopes == "None":
            print('Inside Model, scopes_to_train is ALL TRAINABLE VARIABLES\n\n', flush = True)
            return tf.trainable_variables()
        else:
            scopes_to_train=[scope.strip() for scope in FLAGS.trainable_scopes.split(',')]
            print('Inside Model, scopes_to_train is {}\n\n'.format(scopes_to_train), flush = True)
            
            print('Inside Model, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "classify"): {}\n\n'.format(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'classify')), flush = True)
            
            #as sanity check
            for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'classify'):
                if i not in tf.trainable_variables():
                    print('{} in get_collection but not in tf.trainable_variables'.format(i), flush=True)
            
            for i in tf.trainable_variables():
                if i not in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'classify'):
                    print('{} in tf.trainable_variables but not in get_collection'.format(i), flush=True)
                
            variables_to_train = []
            for v in tf.trainable_variables():
                for scope in scopes_to_train:
                    if scope in v.name.split('/'):
                        variables_to_train.append(v)

            return variables_to_train
        
    
    #hz added Feb19
    def init_fn(self, _, sess):
        sess.run(self.init_op)   
        if FLAGS.load_ckpt != "None":
            print('Inside init_fn, FLAGS.load_ckpt is not "None"', flush = True)
            self.finetuning_saver.restore(sess, FLAGS.load_ckpt)

    @property
    def arg_dir(self):
        return os.path.join(self.train_dir, 'args')

    @property
    def checkpoint_dir(self):
        return os.path.join(self.train_dir, 'tf')

    def train_print(self, text):
        self.tmp.print_queue.append(text)

    def _create_initial_files(self):
        for dir in (self.checkpoint_dir, self.arg_dir):
            if not os.path.exists(dir):
                os.makedirs(dir)
        self.save_args()

    def _reset_files(self):
        shutil.rmtree(self.train_dir)
        self._create_initial_files()

    def save_args(self, **extra_params):
        with open(os.path.join(self.arg_dir, 'args.json'), 'w') as f:
            json.dump({**self.params, **extra_params}, f, sort_keys=True, indent=4)

    @classmethod
    def load(cls, train_dir):
        with open(os.path.join(train_dir, 'args/args.json'), 'r') as f:
            params = json.load(f)
        instance = cls(train_dir=train_dir, **params)
        instance.train_dir = train_dir
        return instance

    def experiment_name(self, **kwargs):
        args = [x + str(y) for x, y in sorted(kwargs.items())]
        return '_'.join([self.__class__.__name__] + args)

    def eval_mode(self, ckpt=None):
        self.session = tf.Session(config=utils.get_config())
        saver = tf.train.Saver()
        if ckpt is None:
            ckpt = utils.find_latest_checkpoint(self.checkpoint_dir)
        else:
            ckpt = os.path.abspath(ckpt)
        saver.restore(self.session, ckpt)
        self.tmp.step = self.session.run(self.step)
        print('Eval model %s at global_step %d' % (self.__class__.__name__, self.tmp.step))
        return self

    def model(self, **kwargs):
        raise NotImplementedError()

    def add_summaries(self, **kwargs):
        raise NotImplementedError()


class ClassifySemi(Model):
    """Semi-supervised classification."""

    def __init__(self, train_dir: str, dataset: data.DataSet, nclass: int, **kwargs):
        self.nclass = nclass
        Model.__init__(self, train_dir, dataset, nclass=nclass, **kwargs)

    def train_step(self, train_session, data_labeled, data_unlabeled):
        raise NotImplementedError('train_step() is not implemented in libml_SuggestedMultitaskAblation/train.py for FS')
#         x, y = self.session.run([data_labeled, data_unlabeled])
        
#         #to record the label of the labeled set and unlabeled set at each step
#         self.batch_labels_dict['labeled'].append(x['label'])
#         self.batch_labels_dict['unlabeled'].append(y['label'])
        
#         #to record the losses and directly save to disk, instead of accessing through tensorboard
        
#         self.tmp.step, labeled_losses_this_step, unlabeled_losses_unscaled_this_step, unlabeled_losses_scaled_this_step, unlabeled_losses_multiplier_this_step = train_session.run([self.ops.train_op, self.ops.update_step, self.ops.labeled_losses, self.ops.unlabeled_losses_unscaled, self.ops.unlabeled_losses_scaled, self.ops.unlabeled_losses_multiplier],
#                                           feed_dict={self.ops.x: x['image'],
#                                                      self.ops.y: y['image'],
#                                                      self.ops.label: x['label'],
#                                                      self.ops.unlabeled_label: y['label']})[1:]
        
# #         print('w_match_this_step is {}'.format(w_match_this_step), flush = True)
        
#         self.losses_dict['labeled_losses'].append(labeled_losses_this_step)
#         self.losses_dict['unlabeled_losses_unscaled'].append(unlabeled_losses_unscaled_this_step)
#         self.losses_dict['unlabeled_losses_scaled'].append(unlabeled_losses_scaled_this_step)
#         self.losses_dict['unlabeled_losses_multiplier'].append(unlabeled_losses_multiplier_this_step)

    def train(self, train_nimg, report_nimg):
        raise NotImplementedError('train() is not implemented in libml_SuggestedMultitaskAblation/train.py for FS')

#         if FLAGS.eval_ckpt:
#             self.eval_checkpoint(FLAGS.eval_ckpt)
#             return
#         batch = FLAGS.batch
#         train_labeled = self.dataset.train_labeled.batch(batch).prefetch(16)
#         train_labeled = train_labeled.make_one_shot_iterator().get_next()
#         train_unlabeled = self.dataset.train_unlabeled.batch(batch).prefetch(16)
#         train_unlabeled = train_unlabeled.make_one_shot_iterator().get_next()
                
                
              
#         scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
#                                                           pad_step_number=10), init_fn=self.init_fn)
            
# #         scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
# #                                                           pad_step_number=10))

#         with tf.Session(config=utils.get_config()) as sess:
#             self.session = sess
#             self.cache_eval()
#         with tf.train.MonitoredTrainingSession(
#                 scaffold=scaffold,
#                 checkpoint_dir=self.checkpoint_dir,
#                 config=utils.get_config(),
#                 save_checkpoint_steps=FLAGS.save_kimg << 10,
#                 save_summaries_steps=report_nimg - batch) as train_session:
#             self.session = train_session._tf_sess()
            
# #             #Sanity check: print out the value of the last fully connected layer
# #             print('###############################################', flush = True)
# #             print(self.session.run('classify/dense/kernel:0'), flush = True)
# #             print('###############################################', flush = True)
# #             print(self.session.run('classify/dense/bias:0'), flush = True)
# #             print('###############################################', flush = True)

# #             if not FLAGS.reset_global_step:
# #                 print('!!!!!!!!!!!!Inside train.py, FLAGS.reset_global_step is FALSE!!!!!!!!!!!!!!!!!')
# #                 self.tmp.step = self.session.run(self.step)
# #             else:
# #                 print('!!!!!!!!!!!!Inside train.py, FLAGS.reset_global_step is TRUE!!!!!!!!!!!!!!!!!')
# #                 self.tmp.step = self.session.run(self.re_initialize_step)
            
#             self.tmp.step = self.session.run(self.step)
#             while self.tmp.step < train_nimg:
#                 loop = trange(self.tmp.step % report_nimg, report_nimg, batch,
#                               leave=False, unit='img', unit_scale=batch,
#                               desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg), train_nimg // report_nimg))
                
#                 #Sanity check: to see what is the initial value, do the parameters updates or not under different FLAGS setting
#                 print('###############################################', flush = True)
# #                 print(self.session.run('classify/dense/kernel:0'), flush = True)
#                 print('###############################################', flush = True)
#                 print(self.session.run('classify/dense/bias:0'), flush = True)
#                 print('###############################################', flush = True)
#                 print('\n\n###############################################', flush = True)
# #                 print(self.session.run('classify/conv2d/kernel:0'), flush = True)
#                 print('###############################################', flush = True)
#                 print(self.session.run('classify/conv2d/bias:0'), flush = True)
#                 print('###############################################', flush = True)
#                 for _ in loop:
#                     self.train_step(train_session, train_labeled, train_unlabeled)
#                     while self.tmp.print_queue:
#                         loop.write(self.tmp.print_queue.pop(0))
                        
# #                 hz_utils.save_pickle(self.train_dir, 'batch_labels_dict_{}.pkl'.format(self.tmp.step), self.batch_labels_dict)
                        
                    
#             while self.tmp.print_queue:
#                 print(self.tmp.print_queue.pop(0))

    def tune(self, train_nimg):
        raise NotImplementedError('tune() is not implemented in libml_SuggestedMultitaskAblation/train.py for FS')

#         batch = FLAGS.batch
#         train_labeled = self.dataset.train_labeled.batch(batch).prefetch(16)
#         train_labeled = train_labeled.make_one_shot_iterator().get_next()
#         train_unlabeled = self.dataset.train_unlabeled.batch(batch).prefetch(16)
#         train_unlabeled = train_unlabeled.make_one_shot_iterator().get_next()

#         for _ in trange(0, train_nimg, batch, leave=False, unit='img', unit_scale=batch, desc='Tuning'):
#             x, y = self.session.run([train_labeled, train_unlabeled])
#             self.session.run([self.ops.tune_op], feed_dict={self.ops.x: x['image'],
#                                                             self.ops.y: y['image'],
#                                                             self.ops.label: x['label']})

    def eval_checkpoint(self, ckpt=None):
        self.eval_mode(ckpt)
        self.cache_eval()
        raw = self.eval_stats(classify_op=self.ops.classify_raw)
        ema = self.eval_stats(classify_op=self.ops.classify_op)
        self.tune(16384)
        tuned_raw = self.eval_stats(classify_op=self.ops.classify_raw)
        tuned_ema = self.eval_stats(classify_op=self.ops.classify_op)
        print('%16s %8s %8s %8s' % ('', 'labeled', 'valid', 'test'))
        print('%16s %8s %8s %8s' % (('raw',) + tuple('%.2f' % x for x in raw)))
        print('%16s %8s %8s %8s' % (('ema',) + tuple('%.2f' % x for x in ema)))
        print('%16s %8s %8s %8s' % (('tuned_raw',) + tuple('%.2f' % x for x in tuned_raw)))
        print('%16s %8s %8s %8s' % (('tuned_ema',) + tuple('%.2f' % x for x in tuned_ema)))

    def cache_eval(self):
        """Cache datasets for computing eval stats."""

        def collect_samples(dataset):
            """Return numpy arrays of all the samples from a dataset."""
            it = dataset.batch(1).prefetch(16).make_one_shot_iterator().get_next()
            images, labels = [], []
            while 1:
                try:
                    v = self.session.run(it)
                except tf.errors.OutOfRangeError:
                    break
                images.append(v['image'])
                labels.append(v['label'])

            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
            return images, labels
    
        def collect_samples_SuggestedMultitaskAblation_LabeledTrain(dataset):
            """Return numpy arrays of all the samples from a dataset."""
            it = dataset.batch(1).prefetch(16).make_one_shot_iterator().get_next()
            images, labels = [], []
            while 1:
                try:
                    v = self.session.run(it)
                except tf.errors.OutOfRangeError:
                    break
                
                images.append(v['image'])
                labels.append(v['label_diagnosis'])
                
            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
            return images, labels
                    
            

        if 'test' not in self.tmp.cache:
            self.tmp.cache.test = collect_samples(self.dataset.test)
            self.tmp.cache.valid = collect_samples(self.dataset.valid)
            self.tmp.cache.train_labeled = collect_samples_SuggestedMultitaskAblation_LabeledTrain(self.dataset.eval_labeled)
        
#         if not self.is_fs:
#             self.tmp.cache.train_unlabeled = collect_samples(self.dataset.eval_unlabeled)
            
    def eval_stats(self, batch=None, feed_extra=None, classify_op=None):
        """Evaluate model on train, valid and test."""
        batch = batch or FLAGS.batch
        classify_op = self.ops.classify_op if classify_op is None else classify_op
        classify_raw = self.ops.classify_raw
        
        accuracies = []
#         subsets = ('train_labeled', 'valid', 'test')
        subsets = ('train_labeled', 'valid', 'test')
        for subset in subsets:
            images, labels = self.tmp.cache[subset]
            
#             To see the loaded test image values 
            if subset == 'test':
                hz_utils.save_pickle(os.path.join(self.train_dir, 'TestImageValuePrintOut'), 'test_image_{}.pkl'.format(self.tmp.step), images) #每evaluate 一次, save 一下目前为止的所有batch_labels_dict
            
                
            predicted = []
            predicted_raw = []

            print('Inside train.py eval_stats, current step is {}'.format(self.tmp.step), flush = True)
            print('Inside train.py eval_stats, current train_dir is {}'.format(self.train_dir), flush = True)
            
            #save predictions:
            predictions_save_dict = dict()
            
            
            for x in range(0, images.shape[0], batch):
                p, p_raw = self.session.run(
                    [classify_op, classify_raw],
                    feed_dict={
                        self.ops.x: images[x:x + batch],
                        **(feed_extra or {})
                    })
                
                predicted.append(p)
                predicted_raw.append(p_raw)
                
            predicted = np.concatenate(predicted, axis=0)
            predicted_raw = np.concatenate(predicted_raw, axis=0)
            
            ema_accuracy_this_step = (predicted.argmax(1) == labels).mean() * 100
            raw_accuracy_this_step = (predicted_raw.argmax(1) == labels).mean() * 100
            
            ema_balanced_accuracy_this_step, ema_class0_recall_this_step, ema_class1_recall_this_step, ema_class2_recall_this_step = hz_utils.calculate_balanced_accuracy(labels, predicted.argmax(1), 'all')
            raw_balanced_accuracy_this_step, raw_class0_recall_this_step, raw_class1_recall_this_step, raw_class2_recall_this_step = hz_utils.calculate_balanced_accuracy(labels, predicted_raw.argmax(1), 'all')
        
            accuracies.append(ema_accuracy_this_step)
            accuracies.append(ema_balanced_accuracy_this_step)
            accuracies.append(ema_class0_recall_this_step)
            accuracies.append(ema_class1_recall_this_step)
            accuracies.append(ema_class2_recall_this_step)
            
            accuracies.append(raw_accuracy_this_step)
            accuracies.append(raw_balanced_accuracy_this_step)
            accuracies.append(raw_class0_recall_this_step)
            accuracies.append(raw_class1_recall_this_step)
            accuracies.append(raw_class2_recall_this_step)
            
            #save predictions to disk for every evaluation
            predictions_save_dict['ema_predictions'] = predicted
            predictions_save_dict['raw_predictions'] = predicted_raw
            predictions_save_dict['true_labels'] = labels
            predictions_save_dict['ema_accuracy'] = ema_accuracy_this_step
            predictions_save_dict['raw_accuracy'] = raw_accuracy_this_step
            
            predictions_save_dict['ema_balanced_accuracy'] = ema_balanced_accuracy_this_step
            predictions_save_dict['ema_class0_recall'] = ema_class0_recall_this_step
            predictions_save_dict['ema_class1_recall'] = ema_class1_recall_this_step
            predictions_save_dict['ema_class2_recall'] = ema_class2_recall_this_step
            
            predictions_save_dict['raw_balanced_accuracy'] = raw_balanced_accuracy_this_step
            predictions_save_dict['raw_class0_recall'] = raw_class0_recall_this_step
            predictions_save_dict['raw_class1_recall'] = raw_class1_recall_this_step
            predictions_save_dict['raw_class2_recall'] = raw_class2_recall_this_step
            
            
            hz_utils.save_pickle(os.path.join(self.train_dir,'predictions'), subset + '_step_' + str(self.tmp.step) + '_predictions.pkl', predictions_save_dict)
            
            if subset == 'valid' and ema_accuracy_this_step > self.best_balanced_validation_accuracy_ema:
                self.best_balanced_validation_accuracy_ema = ema_balanced_accuracy_this_step
                #save checkpoint
                print('Found new record validation_raw!', flush = True)
                self.saver.save(self.session, '{}/best_balanced_validation_accuracy_ema.ckpt'.format(self.train_dir))

            if subset == 'valid' and raw_accuracy_this_step > self.best_balanced_validation_accuracy_raw:
                self.best_balanced_validation_accuracy_raw = raw_balanced_accuracy_this_step
                #save checkpoint
                print('Found new record validation_ema!', flush = True)
                self.saver.save(self.session, '{}/best_balanced_validation_accuracy_raw.ckpt'.format(self.train_dir))

          
        #save the labels of each batch until current epoch
        hz_utils.save_pickle(os.path.join(self.train_dir, 'batch_labels'), 'batch_labels_dict_{}.pkl'.format(self.tmp.step), self.batch_labels_dict) #每evaluate 一次, save 一下目前为止的所有batch_labels_dict
            
        #only retain the last batch_labels_dict
        all_batch_labels_files = [file for file in os.listdir(os.path.join(self.train_dir, 'batch_labels')) if file.startswith('batch_labels')]
        print('At current epoch, all_batch_labels_files are: {}'.format(all_batch_labels_files), flush = True)
        remove_files = [file for file in all_batch_labels_files if file != 'batch_labels_dict_{}.pkl'.format(self.tmp.step)]

        for file in remove_files:
            os.remove(os.path.join(self.train_dir, 'batch_labels', file))
        
        #save the losses of each batch until current epoch
        hz_utils.save_pickle(os.path.join(self.train_dir, 'losses'), 'losses_dict.pkl', self.losses_dict)
        
#         if not self.is_fs:
#             self.train_print('kimg %-5d  accuracy train_labeled_balanced_ema/train_unlabeled_balanced_ema/valid_balanced_ema/test_balanced_ema %.2f  %.2f  %.2f %.2f' % tuple([self.tmp.step >> 10] + [accuracies[1], accuracies[11], accuracies[21], accuracies[31]]))
#         else:
#             self.train_print('kimg %-5d  accuracy train_labeled_balanced_ema/valid_balanced_ema/test_balanced_ema  %.2f  %.2f  %.2f' % tuple([self.tmp.step >> 10] + [accuracies[1], accuracies[11], accuracies[21]]))
        self.train_print('kimg %-5d  accuracy train_labeled_balanced_ema/valid_balanced_ema/test_balanced_ema  %.2f  %.2f  %.2f' % tuple([self.tmp.step >> 10] + [accuracies[1], accuracies[11], accuracies[21]]))
        
        return np.array(accuracies, 'f')

    def add_summaries(self, feed_extra=None, **kwargs):
        del kwargs

        def gen_stats():
            return self.eval_stats(feed_extra=feed_extra)
        
        accuracies = tf.py_func(gen_stats, [], tf.float32)
        
        
        tf.summary.scalar('ema/train_labeled/accuarcy', accuracies[0])
        tf.summary.scalar('ema/train_labeled/balanced_accuarcy', accuracies[1])
        tf.summary.scalar('ema/train_labeled/class0_recall', accuracies[2])
        tf.summary.scalar('ema/train_labeled/class1_recall', accuracies[3])
        tf.summary.scalar('ema/train_labeled/class2_recall', accuracies[4])
        tf.summary.scalar('raw/train_labeled/accuarcy', accuracies[5])
        tf.summary.scalar('raw/train_labeled/balanced_accuarcy', accuracies[6])
        tf.summary.scalar('raw/train_labeled/class0_recall', accuracies[7])
        tf.summary.scalar('raw/train_labeled/class1_recall', accuracies[8])
        tf.summary.scalar('raw/train_labeled/class2_recall', accuracies[9])

#         tf.summary.scalar('ema/train_unlabeled/accuarcy', accuracies[10])
#         tf.summary.scalar('ema/train_unlabeled/balanced_accuarcy', accuracies[11])
#         tf.summary.scalar('ema/train_unlabeled/class0_recall', accuracies[12])
#         tf.summary.scalar('ema/train_unlabeled/class1_recall', accuracies[13])
#         tf.summary.scalar('ema/train_unlabeled/class2_recall', accuracies[14])
#         tf.summary.scalar('raw/train_unlabeled/accuarcy', accuracies[15])
#         tf.summary.scalar('raw/train_unlabeled/balanced_accuarcy', accuracies[16])
#         tf.summary.scalar('raw/train_unlabeled/class0_recall', accuracies[17])
#         tf.summary.scalar('raw/train_unlabeled/class1_recall', accuracies[18])
#         tf.summary.scalar('raw/train_unlabeled/class2_recall', accuracies[19])

        tf.summary.scalar('ema/valid/accuarcy', accuracies[10])
        tf.summary.scalar('ema/valid/balanced_accuarcy', accuracies[11])
        tf.summary.scalar('ema/valid/class0_recall', accuracies[12])
        tf.summary.scalar('ema/valid/class1_recall', accuracies[13])
        tf.summary.scalar('ema/valid/class2_recall', accuracies[14])
        tf.summary.scalar('raw/valid/accuarcy', accuracies[15])
        tf.summary.scalar('raw/valid/balanced_accuarcy', accuracies[16])
        tf.summary.scalar('raw/valid/class0_recall', accuracies[17])
        tf.summary.scalar('raw/valid/class1_recall', accuracies[18])
        tf.summary.scalar('raw/valid/class2_recall', accuracies[19])

        tf.summary.scalar('ema/test/accuarcy', accuracies[20])
        tf.summary.scalar('ema/test/balanced_accuarcy', accuracies[21])
        tf.summary.scalar('ema/test/class0_recall', accuracies[22])
        tf.summary.scalar('ema/test/class1_recall', accuracies[23])
        tf.summary.scalar('ema/test/class2_recall', accuracies[24])
        tf.summary.scalar('raw/test/accuarcy', accuracies[25])
        tf.summary.scalar('raw/test/balanced_accuarcy', accuracies[26])
        tf.summary.scalar('raw/test/class0_recall', accuracies[27])
        tf.summary.scalar('raw/test/class1_recall', accuracies[28])
        tf.summary.scalar('raw/test/class2_recall', accuracies[29])

            
            
            
            
            

