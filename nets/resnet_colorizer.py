from __future__ import print_function

import tensorflow as tf

from .resnet import ResNet


class ResNetColorizer(ResNet):
    def __init__(self, is_training=True, data_format='channels_last', batch_norm_decay=0.997, batch_norm_epsilon=1e-5):
        super(ResNetColorizer, self).__init__(is_training, data_format, batch_norm_decay, batch_norm_epsilon)

    def forward(self, images,  input_data_format='channels_last'):
        # images [1, 3, HEIGHT(256), WIDTH(256), CHANNEL(1)] for exemplar
        # images [1, 1, HEIGHT(128), WIDTH(128), CHANNEL(1)] for search
        # images [BATCH, 4, HEIGHT(256), WIDTH(256), CHANNEL(1)]
        _, _, height, width, _ = images.shape
        images = tf.reshape(images, (-1, height, width, 1))
        tf.summary.image('inputs/images', images, max_outputs=8)

        features = self.feature(images, input_data_format)

        return features


    def feature(self, x, input_data_format='channels_last'):
        # resnet_layer = self._residual_v2
        resnet_layer = self._residual_v2

        assert input_data_format in ('channels_first', 'channels_last')
        if self._data_format != input_data_format:
            if input_data_format == 'channels_last':
                x = tf.transpose(x, [0, 3, 1, 2])
            else:
                x = tf.transpose(x, [0, 2, 3, 1])

        with tf.name_scope('stage0') as name_scope:
            x = x / 128 - 1
            x = self._conv(x, kernel_size=7, filters=64, strides=2)
            x = self._batch_norm(x)
            x = self._relu(x)
            tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())

        with tf.name_scope('stage1'):
            x = resnet_layer(x, kernel_size=3, in_filter=64, out_filter=64, stride=2)
            x = resnet_layer(x, kernel_size=3, in_filter=64, out_filter=64, stride=1)

        with tf.name_scope('stage2'):
            x = resnet_layer(x, kernel_size=3, in_filter=64, out_filter=128, stride=2)
            x = resnet_layer(x, kernel_size=3, in_filter=128, out_filter=128, stride=1)

        with tf.name_scope('stage3'):
            x = resnet_layer(x, kernel_size=3, in_filter=128, out_filter=256, stride=1)
            x = resnet_layer(x, kernel_size=3, in_filter=256, out_filter=256, stride=1)

        with tf.name_scope('stage4'):
            x = resnet_layer(x, kernel_size=3, in_filter=256, out_filter=256, stride=1)
            x = resnet_layer(x, kernel_size=3, in_filter=256, out_filter=256, stride=1)

        with tf.name_scope('feature'):
            x = resnet_layer(x, kernel_size=1, in_filter=256, out_filter=64, stride=1)
            #x = self._conv(x, kernel_size=1, filters=64, strides=1)

        return x
