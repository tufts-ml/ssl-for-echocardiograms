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
"""Classifier architectures."""

import functools
import itertools

import tensorflow as tf
from absl import flags

from libml import layers
from libml.train import ClassifySemi


class ResNet(ClassifySemi):
    def classifier(self, x, scales, filters, repeat, training, getter=None, **kwargs):
        del kwargs
        leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.1)
        bn_args = dict(training=training, momentum=0.999)

        def conv_args(k, f):
            return dict(padding='same',
                        kernel_initializer=tf.random_normal_initializer(stddev=tf.rsqrt(0.5 * k * k * f)))

        def residual(x0, filters, stride=1, activate_before_residual=False):
            x = leaky_relu(tf.layers.batch_normalization(x0, **bn_args))
            if activate_before_residual:
                x0 = x

            x = tf.layers.conv2d(x, filters, 3, strides=stride, **conv_args(3, filters))
            x = leaky_relu(tf.layers.batch_normalization(x, **bn_args))
            x = tf.layers.conv2d(x, filters, 3, **conv_args(3, filters))

            if x0.get_shape()[3] != filters:
                x0 = tf.layers.conv2d(x0, filters, 1, strides=stride, **conv_args(1, filters))

            return x0 + x

        with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
            y = tf.layers.conv2d((x - self.dataset.mean) / self.dataset.std, 16, 3, **conv_args(3, 16))
            for scale in range(scales):
                y = residual(y, filters << scale, stride=2 if scale else 1, activate_before_residual=scale == 0)
                for i in range(repeat - 1):
                    y = residual(y, filters << scale)

            y = leaky_relu(tf.layers.batch_normalization(y, **bn_args))
            y = tf.reduce_mean(y, [1, 2])
            logits = tf.layers.dense(y, self.nclass, kernel_initializer=tf.glorot_normal_initializer())
        return logits


class ResNet_multitask(ClassifySemi):
    def classifier(self, x, scales, filters, repeat, training, getter=None, **kwargs):
        del kwargs
        leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.1)
        bn_args = dict(training=training, momentum=0.999)

        def conv_args(k, f):
            return dict(padding='same',
                        kernel_initializer=tf.random_normal_initializer(stddev=tf.rsqrt(0.5 * k * k * f)))

        def residual(x0, filters, stride=1, activate_before_residual=False):
            x = leaky_relu(tf.layers.batch_normalization(x0, **bn_args))
            if activate_before_residual:
                x0 = x

            x = tf.layers.conv2d(x, filters, 3, strides=stride, **conv_args(3, filters))
            x = leaky_relu(tf.layers.batch_normalization(x, **bn_args))
            x = tf.layers.conv2d(x, filters, 3, **conv_args(3, filters))

            if x0.get_shape()[3] != filters:
                x0 = tf.layers.conv2d(x0, filters, 1, strides=stride, **conv_args(1, filters))

            return x0 + x

        with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
            y = tf.layers.conv2d((x - self.dataset.mean) / self.dataset.std, 16, 3, **conv_args(3, 16))
            for scale in range(scales):
                y = residual(y, filters << scale, stride=2 if scale else 1, activate_before_residual=scale == 0)
                for i in range(repeat - 1):
                    y = residual(y, filters << scale)

            y = leaky_relu(tf.layers.batch_normalization(y, **bn_args))
            y = tf.reduce_mean(y, [1, 2])
            extracted_features = tf.layers.dense(y, self.nclass, kernel_initializer=tf.glorot_normal_initializer())
        
            diagnosis_logits = tf.layers.dense(extracted_features, self.nclass, kernel_initializer=tf.glorot_normal_initializer(), name='diagnosis_logits')
            view_logits = tf.layers.dense(extracted_features, self.nclass, kernel_initializer=tf.glorot_normal_initializer(), name='view_logits')
        return diagnosis_logits, view_logits
    
    



class MultiModel(ResNet, ResNet_multitask):
    MODEL_RESNET, MODEL_RESNET_MULTITASK = 'resnet resnet_multitask'.split()
    MODELS = MODEL_RESNET, MODEL_RESNET_MULTITASK

    def augment(self, x, l, smoothing, **kwargs):
        del kwargs
        return x, l - smoothing * (l - 1. / self.nclass)

    def classifier(self, x, arch, **kwargs):
       
        if arch == self.MODEL_RESNET:
            return ResNet.classifier(self, x, **kwargs)
        elif arch == self.MODEL_RESNET_MULTITASK:
            return ResNet_multitask.classifier(self, x, **kwargs)
        raise ValueError('Model %s does not exists, available ones are %s' % (arch, self.MODELS))


flags.DEFINE_enum('arch', MultiModel.MODEL_RESNET, MultiModel.MODELS, 'Architecture.')
