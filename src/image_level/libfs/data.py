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

import os

from absl import flags
from libml import data, utils

FLAGS = flags.FLAGS


class DataSetFS(data.DataSet):
    @classmethod
    def creator(cls, name, train_files, valid_files, test_files, augment, parse_fn=data.default_parse, memoize_fn=data.memoize,
                colors=3, nclass=3, height=32, width=32):
        

        train_files = [os.path.join(data.DATA_DIR, x) for x in train_files]
        valid_files = [os.path.join(data.DATA_DIR, x) for x in valid_files]
        test_files = [os.path.join(data.DATA_DIR, x) for x in test_files]

#         fn = data.memoize if do_memoize else lambda x: x.repeat().shuffle(FLAGS.shuffle)
        fn = memoize_fn

        def create():
            para = max(1, len(utils.get_available_gpus())) * FLAGS.para_augment
            
            train_labeled = parse_fn(data.dataset(train_files))
            valid = parse_fn(data.dataset(valid_files))
            test = parse_fn(data.dataset(test_files))
            
#             print('Using {} as labeled train set'.format(train_files), flush = True)
#             print('Using {} as validation set'.format(valid_files), flush = True)
#             print('Using {} as test set'.format(test_files), flush = True)
                    
            if FLAGS.whiten:
                mean, std = data.compute_mean_std(train_labeled)
            else:
                mean, std = 0, 1

            return cls(name,
                       train_labeled=fn(train_labeled).map(augment, para),
                       train_unlabeled=None,
#                        eval_labeled=train_labeled.take(5000),  # No need to to eval on everything.
                       eval_labeled=train_labeled,
                       eval_unlabeled=None,
                       valid=valid,
                       test=test,
                       nclass=nclass, colors=colors, height=height, width=width, mean=mean, std=std)

        return name, create



