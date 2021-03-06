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

import tensorflow as tf
from absl import flags
from tqdm import trange

from libml import data, utils
from libml.train import ClassifySemi

FLAGS = flags.FLAGS


class ClassifyFullySupervised(ClassifySemi):
    
    """Fully supervised classification.
    """

    def __init__(self, train_dir: str, dataset: data.DataSet, nclass: int, **kwargs):
        ClassifySemi.__init__(self, train_dir, dataset, nclass, **kwargs)
        self.losses_dict = {'total_loss':[], 'unweighted_diagnosis_loss':[], 'weighted_diagnosis_loss':[], 'unweighted_view_loss':[], 'weighted_view_loss':[], 'scaled_weighted_view_loss':[]}


    
    def train_step(self, train_session, data_labeled):
        x = self.session.run(data_labeled)

        self.tmp.step, total_loss_this_step, unweighted_diagnosis_loss_this_step, weighted_diagnosis_loss_this_step, unweighted_view_loss_this_step, weighted_view_loss_this_step, scaled_weighted_view_loss_this_step = train_session.run([self.ops.train_op, self.ops.update_step, self.ops.total_loss, self.ops.unweighted_diagnosis_loss, self.ops.weighted_diagnosis_loss, self.ops.unweighted_view_loss, self.ops.weighted_view_loss, self.ops.scaled_weighted_view_loss],
                                          feed_dict={self.ops.x: x['image'],
                                                     self.ops.label_diagnosis: x['label_diagnosis'],
                                                     self.ops.label_view: x['label_view']})[1:]
        
        
        self.losses_dict['total_loss'].append(total_loss_this_step)
        self.losses_dict['unweighted_diagnosis_loss'].append(unweighted_diagnosis_loss_this_step)
        self.losses_dict['weighted_diagnosis_loss'].append(weighted_diagnosis_loss_this_step)
        self.losses_dict['unweighted_view_loss'].append(unweighted_view_loss_this_step)
        self.losses_dict['weighted_view_loss'].append(weighted_view_loss_this_step)
        self.losses_dict['scaled_weighted_view_loss'].append(scaled_weighted_view_loss_this_step)


    def train(self, train_nimg, report_nimg):
        if FLAGS.eval_ckpt:
            self.eval_checkpoint(FLAGS.eval_ckpt)
            return
        batch = FLAGS.batch
        train_labeled = self.dataset.train_labeled.batch(batch).prefetch(16)
        train_labeled = train_labeled.make_one_shot_iterator().get_next()
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
                                                          pad_step_number=10))

        with tf.Session(config=utils.get_config()) as sess:
            self.session = sess
            self.cache_eval()

        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=self.checkpoint_dir,
                config=utils.get_config(),
                save_checkpoint_steps=FLAGS.save_kimg << 10,
                save_summaries_steps=report_nimg - batch) as train_session:
            self.session = train_session._tf_sess()
            self.tmp.step = self.session.run(self.step)
        
            while self.tmp.step < train_nimg:
                loop = trange(self.tmp.step % report_nimg, report_nimg, batch,
                              leave=False, unit='img', unit_scale=batch,
                              desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg), train_nimg // report_nimg))
                
                for _ in loop:
                    self.train_step(train_session, train_labeled)
                    while self.tmp.print_queue:
                        loop.write(self.tmp.print_queue.pop(0))
            while self.tmp.print_queue:
                print(self.tmp.print_queue.pop(0))

    def tune(self, train_nimg):
        batch = FLAGS.batch
        train_labeled = self.dataset.train_labeled.batch(batch).prefetch(16)
        train_labeled = train_labeled.make_one_shot_iterator().get_next()

        for _ in trange(0, train_nimg, batch, leave=False, unit='img', unit_scale=batch, desc='Tuning'):
            x = self.session.run([train_labeled])
            self.session.run([self.ops.tune_op], feed_dict={self.ops.x: x['image'],
                                                            self.ops.label_diagnosis: x['label_diagnosis'],
                                                            self.ops.label_view: x['label_view']})
    
