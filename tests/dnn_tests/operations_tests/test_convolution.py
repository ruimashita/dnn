# -*- coding: utf-8 -*-
import numpy as np

from dnn.operations.convolution import (
    Convolution2D_IM2COL, Convolution2D_Naive,
    Convolution2D_KN2ROW, Convolution2D_KN2COL,
)
from dnn.test import check_gradients


def test_compare_convolution2d():
    input_shape = (1, 32, 20, 20)  # b, c, h, w
    input_data = np.random.rand(*input_shape)

    kernel_shape = (64, 32, 3, 3)  # out_c, in_c, h, w,
    kernel_data = np.random.rand(*kernel_shape)

    pad = 1
    stride = 4

    naive = Convolution2D_Naive(pad=pad, stride=stride)
    naive_output = naive(input_data, kernel_data)

    kn2row = Convolution2D_KN2ROW(pad=pad, stride=stride)
    kn2row_output = kn2row(input_data, kernel_data)

    assert np.allclose(naive_output.data, kn2row_output.data)

    kn2col = Convolution2D_KN2COL(pad=pad, stride=stride)
    kn2col_output = kn2col(input_data, kernel_data)

    assert np.allclose(naive_output.data, kn2col_output.data)

    im2col = Convolution2D_IM2COL(pad=pad, stride=stride)
    im2col_output = im2col(input_data, kernel_data)

    assert np.allclose(naive_output.data, im2col_output.data)


def test_convolution2d_im2col():
    input_shape = (2, 5, 7, 7)  # b, c, h, w
    input_data = np.random.rand(*input_shape)

    kernel_shape = (6, 5, 3, 3)  # out_c, in_c, h, w,
    kernel_data = np.random.rand(*kernel_shape)

    op = Convolution2D_IM2COL(pad=1, stride=2)

    theorical_jacobians, numerical_jacobians = check_gradients(op, (input_data, kernel_data))
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian, atol=1e-06), \
            "theorical: {}, \n numerical: {}".format(theorical_jacobian, numerical_jacobian)


def test_convolution2d_naive():
    input_shape = (2, 5, 7, 7)  # b, c, h, w
    input_data = np.random.rand(*input_shape)

    kernel_shape = (6, 5, 3, 3)  # out_c, in_c, h, w,
    kernel_data = np.random.rand(*kernel_shape)

    op = Convolution2D_Naive(pad=1, stride=1)

    theorical_jacobians, numerical_jacobians = check_gradients(op, (input_data, kernel_data))
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian, atol=1e-06), \
            "theorical: {}, \n numerical: {}".format(theorical_jacobian, numerical_jacobian)


if __name__ == '__main__':
    test_compare_convolution2d()
    test_convolution2d_im2col()
    test_convolution2d_naive()
