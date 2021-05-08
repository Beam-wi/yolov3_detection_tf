# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

from utils import common
from utils.mnv3_layers import *

def conv2d(inputs, filters, kernel_size, strides=1):
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs
    if strides > 1: 
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs

def darknet53_body(inputs, train_with_two_feature_map=False):
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1)
        net = conv2d(net, filters * 2, 3)
        net = net + shortcut

        return net
    
    # first two conv2d layers
    net = conv2d(inputs, 32,  3, strides=1)
    net = conv2d(net, 64,  3, strides=2)

    # res_block * 1
    net = res_block(net, 32)

    net = conv2d(net, 128, 3, strides=2)

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)

    net = conv2d(net, 256, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 128)

    route_1 = net
    net = conv2d(net, 512, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 256)

    route_2 = net
    if train_with_two_feature_map:
        return route_1, route_2
    net = conv2d(net, 1024, 3, strides=2)

    # res_block * 4
    for i in range(4):
        net = res_block(net, 512)
    route_3 = net

    return route_1, route_2, route_3


def darknet53_body_prun(inputs, train_with_two_feature_map=False):
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1)
        net = conv2d(net, filters * 2, 3)
        net = net + shortcut

        return net

    prun_rate = 0.5
    # first two conv2d layers
    net = conv2d(inputs, int(32 * prun_rate), 3, strides=1)
    net = conv2d(net, int(64 * prun_rate), 3, strides=2)

    # res_block * 1
    net = res_block(net, int(32 * prun_rate))

    net = conv2d(net, int(128 * prun_rate), 3, strides=2)

    # res_block * 2
    for i in range(2):
        net = res_block(net, int(64 * prun_rate))

    net = conv2d(net, int(256 * prun_rate), 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, int(128 * prun_rate))

    route_1 = net
    net = conv2d(net, int(512 * prun_rate), 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, int(256 * prun_rate))

    route_2 = net
    if train_with_two_feature_map:
        return route_1, route_2
    net = conv2d(net, int(1024 * prun_rate), 3, strides=2)

    # res_block * 4
    for i in range(4):
        net = res_block(net, int(512 * prun_rate))
    route_3 = net

    return route_1, route_2, route_3



def yolo_block(inputs, filters):
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    route = net
    net = conv2d(net, filters * 2, 3)
    return route, net


def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), align_corners=True, name='upsampled')
    return inputs


def mobilenetv2(input_data, train_with_two_feature_map, trainable=True):
    with tf.variable_scope('mobilenetv2'):
        # input_data = tf.reshape(input_data, [-1, 416, 416, 3]) # print layer's shape

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 3, 32), trainable=trainable, name='conv0',
                                          downsample=True)
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32, 16), trainable=trainable, name='conv1',
                                          downsample=True)
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 16, 24), trainable=trainable, name='conv2')

        for i in range(1):
            input_data = common.residual_block(input_data, 24, 24, 24, trainable=trainable, name='residual%d' % (i + 0))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 24, 32), trainable=trainable, name='conv4',
                                          downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 32, 32, 32, trainable=trainable, name='residual%d' % (i + 1))

        route_1 = input_data

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32, 64), trainable=trainable, name='conv7',
                                          downsample=True)

        for i in range(3):
            input_data = common.residual_block(input_data, 64, 384, 64, trainable=trainable,
                                               name='residual%d' % (i + 3))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 64, 96), trainable=trainable, name='conv11')

        for i in range(2):
            input_data = common.residual_block(input_data, 96, 576, 96, trainable=trainable,
                                               name='residual%d' % (i + 6))

        route_2 = input_data

        if train_with_two_feature_map:
            return route_1, route_2

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 96, 160), trainable=trainable, name='conv14',
                                          downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 160, 160, 160, trainable=trainable,
                                               name='residual%d' % (i + 8))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 160, 320), trainable=trainable,
                                          name='conv17')

        return route_1, route_2, input_data


def mobilenetv3(input_data, train_with_two_feature_map=False, trainable=True):
    reduction_ratio = 4
    zoom_factor = 1.5
    with tf.variable_scope('mobilenetv3'):
        # input_data = tf.reshape(input_data, [-1, 416, 416, 3])  # print layer's shape

        input_data = conv2d_block(input_data, 16, 3, 2, trainable, name='conv1_1', h_swish=True)  # size/4 512x512x16

        input_data = mnv3_block(input_data, 3, 16, 16, 1, trainable, name='bneck2_1', h_swish=False, # 512x512x16
                                ratio=reduction_ratio, se=False)

        input_data = mnv3_block(input_data, 3, 64, 24, 2, trainable, name='bneck3_1', h_swish=False,
                                ratio=reduction_ratio, se=False)  # size/4  256x256x24
        input_data = mnv3_block(input_data, 3, 72, 24, 1, trainable, name='bneck3_2', h_swish=False,
                                ratio=reduction_ratio, se=False) # 256x256x24

        input_data = mnv3_block(input_data, 5, 72, 40, 2, trainable, name='bneck4_1', h_swish=False,
                                ratio=reduction_ratio, se=True)  # size/8 128x128x40
        input_data = mnv3_block(input_data, 5, 120, 40, 1, trainable, name='bneck4_2', h_swish=False,
                                ratio=reduction_ratio, se=True) # 128x128x40
        input_data = mnv3_block(input_data, 5, 120, 40, 1, trainable, name='bneck4_3', h_swish=False,
                                ratio=reduction_ratio, se=True) # 128x128x40

        route_1 = input_data

        input_data = mnv3_block(input_data, 3, 240, 80, 2, trainable, name='bneck5_1', h_swish=True,
                                ratio=reduction_ratio, se=False)  # size/16 64x64x80
        input_data = mnv3_block(input_data, 3, 200, 80, 1, trainable, name='bneck5_2', h_swish=True,
                                ratio=reduction_ratio, se=False) # 64x64x80
        input_data = mnv3_block(input_data, 3, 184, 80, 1, trainable, name='bneck5_3', h_swish=True,
                                ratio=reduction_ratio, se=False) # 64x64x80
        input_data = mnv3_block(input_data, 3, 184, 80, 1, trainable, name='bneck5_4', h_swish=True,
                                ratio=reduction_ratio, se=False) # 64x64x80

        input_data = mnv3_block(input_data, 3, 480, 112, 1, trainable, name='bneck6_1', h_swish=True,
                                ratio=reduction_ratio, se=True)  # 64x64x112
        input_data = mnv3_block(input_data, 3, 672, 112, 1, trainable, name='bneck6_2', h_swish=True,
                                ratio=reduction_ratio, se=True)  # 64x64x112

        route_2 = input_data

        if train_with_two_feature_map:
            return route_1, route_2

        input_data = mnv3_block(input_data, 5, 672, 160, 2, trainable, name='bneck7_1', h_swish=True,
                                ratio=reduction_ratio, se=True)  # size/32 32x32x160
        input_data = mnv3_block(input_data, 5, 960, 160, 1, trainable, name='bneck7_2', h_swish=True,
                                ratio=reduction_ratio, se=True) # 32x32x160
        input_data = mnv3_block(input_data, 5, 960, 160, 1, trainable, name='bneck7_3', h_swish=True,
                                ratio=reduction_ratio, se=True) # 32x32x160

        return route_1, route_2, input_data


def mobilenetv3_add_zoom_factor(input_data, train_with_two_feature_map=False, trainable=True, zoom_factor=1.5):
    reduction_ratio = 4
    with tf.variable_scope('mobilenetv3_add_zoom_factor'):
        # input_data = tf.reshape(input_data, [-1, 416, 416, 3])  # print layer's shape

        input_data = conv2d_block(input_data, int(zoom_factor*16), 3, 2, trainable, name='conv1_1', h_swish=True)  # size/4 512x512x16

        input_data = mnv3_block(input_data, 3, int(zoom_factor*16), int(zoom_factor*16), 1, trainable, name='bneck2_1', h_swish=False, # 512x512x16
                                ratio=reduction_ratio, se=False)

        input_data = mnv3_block(input_data, 3, int(zoom_factor*64), int(zoom_factor*24), 2, trainable, name='bneck3_1', h_swish=False,
                                ratio=reduction_ratio, se=False)  # size/4  256x256x24
        input_data = mnv3_block(input_data, 3, int(zoom_factor*72), int(zoom_factor*24), 1, trainable, name='bneck3_2', h_swish=False,
                                ratio=reduction_ratio, se=False) # 256x256x24

        input_data = mnv3_block(input_data, 5, int(zoom_factor*72), int(zoom_factor*40), 2, trainable, name='bneck4_1', h_swish=False,
                                ratio=reduction_ratio, se=True)  # size/8 128x128x40
        input_data = mnv3_block(input_data, 5, int(zoom_factor*120), int(zoom_factor*40), 1, trainable, name='bneck4_2', h_swish=False,
                                ratio=reduction_ratio, se=True) # 128x128x40
        input_data = mnv3_block(input_data, 5, int(zoom_factor*120), int(zoom_factor*40), 1, trainable, name='bneck4_3', h_swish=False,
                                ratio=reduction_ratio, se=True) # 128x128x40

        route_1 = input_data

        input_data = mnv3_block(input_data, 3, int(zoom_factor*240), int(zoom_factor*80), 2, trainable, name='bneck5_1', h_swish=True,
                                ratio=reduction_ratio, se=False)  # size/16 64x64x80
        input_data = mnv3_block(input_data, 3, int(zoom_factor*200), int(zoom_factor*80), 1, trainable, name='bneck5_2', h_swish=True,
                                ratio=reduction_ratio, se=False) # 64x64x80
        input_data = mnv3_block(input_data, 3, int(zoom_factor*184), int(zoom_factor*80), 1, trainable, name='bneck5_3', h_swish=True,
                                ratio=reduction_ratio, se=False) # 64x64x80
        input_data = mnv3_block(input_data, 3, int(zoom_factor*184), int(zoom_factor*80), 1, trainable, name='bneck5_4', h_swish=True,
                                ratio=reduction_ratio, se=False) # 64x64x80

        input_data = mnv3_block(input_data, 3, int(zoom_factor*480), int(zoom_factor*112), 1, trainable, name='bneck6_1', h_swish=True,
                                ratio=reduction_ratio, se=True)  # 64x64x112
        input_data = mnv3_block(input_data, 3, int(zoom_factor*672), int(zoom_factor*112), 1, trainable, name='bneck6_2', h_swish=True,
                                ratio=reduction_ratio, se=True)  # 64x64x112

        route_2 = input_data

        if train_with_two_feature_map:
            return route_1, route_2

        input_data = mnv3_block(input_data, 5, int(zoom_factor*672), int(zoom_factor*160), 2, trainable, name='bneck7_1', h_swish=True,
                                ratio=reduction_ratio, se=True)  # size/32 32x32x160
        input_data = mnv3_block(input_data, 5, int(zoom_factor*960), int(zoom_factor*160), 1, trainable, name='bneck7_2', h_swish=True,
                                ratio=reduction_ratio, se=True) # 32x32x160
        input_data = mnv3_block(input_data, 5, int(zoom_factor*960), int(zoom_factor*160), 1, trainable, name='bneck7_3', h_swish=True,
                                ratio=reduction_ratio, se=True) # 32x32x160

        return route_1, route_2, input_data
