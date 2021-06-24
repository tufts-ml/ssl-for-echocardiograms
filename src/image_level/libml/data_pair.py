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

import itertools

from absl import flags
from libml.data import DataSet, augment_echo
import tensorflow as tf

flags.DEFINE_integer('nu', 2, 'Number of augmentations for class-consistency.')
FLAGS = flags.FLAGS


def stack_augment(augment):
    def func(x):
        xl = [augment(x) for _ in range(FLAGS.nu)]

        return dict(image=tf.stack([x['image'] for x in xl]),
                    label=tf.stack([x['label'] for x in xl]))

    return func

