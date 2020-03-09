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

from blueoil.networks.classification.base import Base
from blueoil.layers import fully_connected, conv2d, max_pooling2d

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg13Network(Base):
    def __init__(
            self,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None, 
            *args, 
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def base(self, images, is_training):

        self.images=images

        keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))

        self.input = self.convert_rbg_to_bgr(images)

        self.conv1 = self.conv_layer("conv1", self.input, filters=64, kernel_size=3)
        self.conv2 = self.conv_layer("conv2", self.conv1, filters=64, kernel_size=3)

        self.pool1 = self.max_pool("pool1", self.conv2, pool_size=2, strides=2)

        self.conv3 = self.conv_layer("conv3", self.pool1, filters=128, kernel_size=3)
        self.conv4 = self.conv_layer("conv4", self.conv3, filters=128, kernel_size=3)

        self.pool2 = self.max_pool("pool2", self.conv4, pool_size=2, strides=2)

        self.conv5 = self.conv_layer("conv5", self.pool2, filters=256, kernel_size=3)
        self.conv6 = self.conv_layer("conv6", self.conv5, filters=256, kernel_size=3)

        self.pool3 = self.max_pool("pool3", self.conv6, pool_size=2, strides=2)

        self.conv7 = self.conv_layer("conv7", self.pool3, filters=512, kernel_size=3)
        self.conv8 = self.conv_layer("conv8", self.conv7, filters=512, kernel_size=3)

        self.pool4 = self.max_pool("pool4", self.conv8, pool_size=2, strides=2)

        self.conv9 = self.conv_layer("conv9", self.pool4, filters=512, kernel_size=3)
        self.conv10 = self.conv_layer("conv10", self.conv9, filters=512, kernel_size=3)

        self.pool5 = self.max_pool("pool5", self.conv10, pool_size=2, strides=2)

        fc11 = self.fc_layer("fc11", self.pool5, filters=4096, activation=tf.nn.relu)
        self.fc11 = tf.nn.dropout(fc11, keep_prob)

        fc12 = self.fc_layer("fc12", self.fc11, filters=4096, activation=tf.nn.relu)
        self.fc12 = tf.nn.dropout(fc12, keep_prob)

        self.fc13 = self.fc_layer("fc13", self.fc12, filters=self.num_classes, activation=None)

        return self.fc13

    def conv_layer(
        self,
        name,
        inputs,
        filters,
        kernel_size,
        strides=1,
        padding="SAME",
        activation=tf.nn.sigmoid,
        *args,
        **kwargs
    ):
        kernel_initializer=tf.contrib.layers.xavier_initializer()
        #biases_initializer=tf.zeros_initializer()

        output = conv2d(
            name=name,
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            kernel_initializer=kernel_initializer,
            #biases_initializer=biases_initializer,
            *args,
            **kwargs
        )

        return output

    def fc_layer(
            self,
            name,
            inputs,
            filters,
            activation,
            *args,
            **kwargs
    ):

        output = fully_connected(
            name=name,
            inputs=inputs,
            filters=filters,
            activation=activation,
            *args,
            **kwargs
        )

        return output

    def max_pool(
            self,
            name,
            inputs,
            pool_size,
            strides,
            *args,
            **kwargs
    ):
        
         output = max_pooling2d(
            name=name,
            inputs=inputs,
            pool_size=pool_size,
            strides=strides,
            *args,
            **kwargs
         )

         return output

    def convert_rbg_to_bgr(self, rgb_images):
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_images)

        bgr_images = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        return bgr_images
