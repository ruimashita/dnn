# -*- coding: utf-8 -*-
import unittest
import math

import numpy as np

from dnn.operations.activation import Relu, Sigmoid, Softmax
from dnn.test import check_gradients


def test_softmax():
    input_shape = (2, 3)
    input_data = np.random.rand(*input_shape)

    op = Softmax(axis=1)

    theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian), \
            "theorical: {}, \n numerical: {}".format(theorical_jacobian, numerical_jacobian)


class TestRelu(unittest.TestCase):

    def test_backward(self):
        input_shape = (2, 3, 4)
        input_data = np.random.rand(*input_shape)

        op = Relu()

        theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
        for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
            assert np.allclose(theorical_jacobian, numerical_jacobian), \
                "theorical: {}, \n numerical: {}".format(theorical_jacobian, numerical_jacobian)


class TestSigmoid(unittest.TestCase):

    def test_forward(self):
        input_data = np.array([-1e+10, -1, 0, 1, 1e+10])
        expect = np.array([0, 1/(1+math.exp(1)), 0.5, 1/(1+math.exp(-1)), 1])
        op = Sigmoid()
        result = op.forward(input_data)

        assert np.allclose(expect, result)

    def test_backward(self):
        input_shape = (2, 3, 4)
        input_data = np.random.rand(*input_shape)

        op = Sigmoid()

        theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
        for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
            assert np.allclose(theorical_jacobian, numerical_jacobian), \
                "theorical: {}, \n numerical: {}".format(theorical_jacobian, numerical_jacobian)


if __name__ == '__main__':
    test_softmax()
