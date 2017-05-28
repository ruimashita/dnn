# -*- coding: utf-8 -*-
import unittest

import numpy as np

from dnn.operations.activation import Relu, Softmax
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

        print(op.forward(input_data))

        theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
        for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
            assert np.allclose(theorical_jacobian, numerical_jacobian), \
                "theorical: {}, \n numerical: {}".format(theorical_jacobian, numerical_jacobian)


if __name__ == '__main__':
    test_softmax()
