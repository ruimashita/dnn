# -*- coding: utf-8 -*-
import math
import unittest

import numpy as np
import tensorflow as tf
from dnn.operations import convolution2d

from examples.mnist_softmax import mnist_data, mnist_softmax
from examples.iris_mlp import iris_data, mlp_training


def test_compare_convolution2d():
    input_shape = (2, 3, 9, 9)  # b, c, h, w
    input_data = np.random.rand(*input_shape)

    kernel_shape = (4, 3, 3, 3)  # out_c, in_c, h, w,
    kernel_data = np.random.rand(*kernel_shape)

    tf.enable_eager_execution()

    tf_out = tf.nn.conv2d(input_data.transpose((0, 2, 3, 1)),
                          kernel_data.transpose((2, 3, 1, 0)),
                          strides=[1, 2, 2, 1],
                          padding="SAME")
    tf_out = np.array(tf_out).transpose((0, 3, 1, 2))
    out = convolution2d(input_data, kernel_data, stride=2, pad=1)

    #    assert np.allclose(tf_out, out.data)
    print(tf_out, out.data)


def test_compare_dilated_convolution2d():
    input_shape = (2, 2, 22, 22)  # b, c, h, w
    input_data = np.random.rand(*input_shape)

    kernel_shape = (3, 2, 3, 3)  # out_c, in_c, h, w,
    kernel_data = np.random.rand(*kernel_shape)

    pad = 2
    stride = 3
    dilation = 2

    tf.enable_eager_execution()

    tf_out = tf.nn.conv2d(input_data.transpose((0, 2, 3, 1)),
                          kernel_data.transpose((2, 3, 1, 0)),
                          strides=[1, stride, stride, 1],
                          dilations=[1, dilation, dilation, 1],
                          padding="SAME")
    tf_out = np.array(tf_out).transpose((0, 3, 1, 2))
    print(tf_out.shape)

    out = convolution2d(input_data, kernel_data, stride=stride, pad=pad, dilation=dilation)

    assert np.allclose(tf_out, out.data)


if __name__ == "__main__":
    test_compare_convolution2d()
    test_compare_dilated_convolution2d()
