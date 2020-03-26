# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import functools

import tensorflow as tf

from blueoil.blocks import lmnet_block
from blueoil.networks.classification.base import Base


class VGG13(Base):
    """Lmnet v1 for classification.
    """
    version = 1.0

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.activation = tf.nn.relu
        self.custom_getter = None

    def _get_lmnet_block(self, is_training, channels_data_format):
        return functools.partial(lmnet_block,
                                 activation=self.activation,
                                 custom_getter=self.custom_getter,
                                 is_training=is_training,
                                 is_debug=self.is_debug,
                                 use_bias=False,
                                 data_format=channels_data_format)

    def _space_to_depth(self, inputs=None, block_size=2, name=''):
        if self.data_format != 'NHWC':
            inputs = tf.transpose(inputs, perm=[self.data_format.find(d) for d in 'NHWC'])

        output = tf.space_to_depth(inputs, block_size=block_size, name=name)

        if self.data_format != 'NHWC':
            output = tf.transpose(output, perm=['NHWC'.find(d) for d in self.data_format])
        return output

    def bn(self, name, l, training):
        return tf.contrib.layers.batch_norm(l,
                                            decay=0.999,
                                            updates_collections=None,
                                            is_training=training,
                                            trainable=True,
                                            center=True,
                                            scale=True,
                                            data_format='NHWC',
                                            scope=name)
        # return tf.layers.batch_normalization(
        #     l,
        #     axis=-1,
        #     momentum=0.997,
        #     epsilon=0.001,
        #     center=True,
        #     scale=True,
        #     beta_initializer=tf.zeros_initializer(),
        #     gamma_initializer=tf.ones_initializer(),
        #     moving_mean_initializer=tf.zeros_initializer(),
        #     moving_variance_initializer=tf.ones_initializer(),
        #     training=training,
        #     trainable=True,
        #     name=name,
        #     reuse=None,
        #     renorm=False,
        #     renorm_clipping=None,
        #     renorm_momentum=0.9,
        #     fused=True)

    def pooling(self, name, x, ks, st):
        return tf.compat.v1.layers.max_pooling2d(x, ks, st, 'same', name=name)
        # return tf.nn.max_pool(x, (1, ks, ks, 1), (1, st, st, 1), 'VALID', name=name)

    def conv(self, name, x, filters, kernel_size, training, strides=(1, 1)):
        x = tf.layers.conv2d(
            x, filters, kernel_size,
            padding="SAME",
            data_format='channels_last',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            strides=strides,
            kernel_regularizer=None,
            bias_regularizer=None,
            use_bias=False,
            activation=None,
            name=name)

        x = self.bn(name + '_bn', x, training)
        return tf.nn.relu(x)

    def base(self, images, is_training, *args, **kwargs):
        """Base network.

        Args:
            images: Input images.
            is_training: A flag for if is training.
        Returns:
            tf.Tensor: Inference result.
        """

        self.images = images

        x = self.conv("conv1", images, 64, 3, is_training)
        x = self.conv("conv2", x, 64, 3, is_training)

        x = self.pooling("pool1", x, 2, 2)

        x = self.conv("conv3", x, 128, 3, is_training)
        x = self.conv("conv4", x, 128, 3, is_training)

        x = self.pooling("pool2", x, 2, 2)

        x = self.conv("conv5", x, 256, 3, is_training)
        x = self.conv("conv6", x, 256, 3, is_training)

        x = self.pooling("pool3", x, 2, 2)

        x = self.conv("conv7", x, 512, 3, is_training)
        x = self.conv("conv8", x, 512, 3, is_training)

        x = self.pooling("pool4", x, 2, 2)

        x = self.conv("conv9", x, 512, 3, is_training)
        x = self.conv("conv10", x, 512, 3, is_training)
        #
        # self._heatmap_layer = x
        #
        x = self.pooling("pool5", x, 2, 2)

        x = tf.compat.v1.layers.flatten(x)

        x = tf.compat.v1.layers.dense(x, 4096,
                                      activation=tf.nn.relu,
                                      use_bias=True,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                      name='fc11')

        x = tf.layers.dropout(x, training=is_training)

        x = tf.compat.v1.layers.dense(x, 4096,
                                      activation=tf.nn.relu,
                                      use_bias=True,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                      name='fc12')

        x = tf.layers.dropout(x, training=is_training)

        x = tf.compat.v1.layers.dense(x, self.num_classes,
                                      activation=None,
                                      use_bias=True,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                      name='fc13')
        return x


class VGG13Quantize(VGG13):
    """Lmnet quantize network for classification, version 1.0

    Following `args` are used for inference: ``activation_quantizer``, ``activation_quantizer_kwargs``,
    ``weight_quantizer``, ``weight_quantizer_kwargs``.

    Args:
        activation_quantizer (callable): Weight quantizater. See more at `lmnet.quantizations`.
        activation_quantizer_kwargs (dict): Kwargs for `activation_quantizer`.
        weight_quantizer (callable): Activation quantizater. See more at `lmnet.quantizations`.
        weight_quantizer_kwargs (dict): Kwargs for `weight_quantizer`.
    """
    version = 1.0

    def __init__(
            self,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        assert weight_quantizer
        assert activation_quantizer

        activation_quantizer_kwargs = activation_quantizer_kwargs if activation_quantizer_kwargs is not None else {}
        weight_quantizer_kwargs = weight_quantizer_kwargs if weight_quantizer_kwargs is not None else {}

        self.activation = activation_quantizer(**activation_quantizer_kwargs)
        weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        self.custom_getter = functools.partial(self._quantized_variable_getter,
                                               weight_quantization=weight_quantization)

    @staticmethod
    def _quantized_variable_getter(getter, name, weight_quantization=None, *args, **kwargs):
        """Get the quantized variables.

        Use if to choose or skip the target should be quantized.

        Args:
            getter: Default from tensorflow.
            name: Default from tensorflow.
            weight_quantization: Callable object which quantize variable.
            args: Args.
            kwargs: Kwargs.
        """
        assert callable(weight_quantization)
        var = getter(name, *args, **kwargs)
        with tf.variable_scope(name):
            # Apply weight quantize to variable whose last word of name is "kernel".
            if "kernel" == var.op.name.split("/")[-1]:
                return weight_quantization(var)
        return var
