# -*- coding: utf-8 -*-
import unittest
import math

import numpy as np

from dnn.operations.convolution import Convolution2DIM2COL, Convolution2DNaive
from dnn.test import check_gradients


def test_convolution2d_im2col():
    input_shape = (2, 5, 7, 7)  # b, h, w, c
    input_data = np.random.rand(*input_shape)

    kernel_shape = (6, 5, 3, 3)  # out_c, h, w, in_c,
    kernel_data = np.random.rand(*kernel_shape)

    op = Convolution2DIM2COL(pad=1, stride=2)

    theorical_jacobians, numerical_jacobians = check_gradients(op, (input_data, kernel_data))
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian, atol=1e-06), \
            "theorical: {}, \n numerical: {}".format(theorical_jacobian, numerical_jacobian)


def test_convolution2d_naive():
    input_shape = (2, 5, 7, 7)  # b, h, w, c
    input_data = np.random.rand(*input_shape)

    kernel_shape = (6, 5, 3, 3)  # out_c, h, w, in_c,
    kernel_data = np.random.rand(*kernel_shape)

    op = Convolution2DNaive(pad=1, stride=1)

    theorical_jacobians, numerical_jacobians = check_gradients(op, (input_data, kernel_data))
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian, atol=1e-06), \
            "theorical: {}, \n numerical: {}".format(theorical_jacobian, numerical_jacobian)


if __name__ == '__main__':
    test_convolution2d_im2col()
    test_convolution2d_naive()
