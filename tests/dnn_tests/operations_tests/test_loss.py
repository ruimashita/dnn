# -*- coding: utf-8 -*-
import numpy as np

from dnn.operations.loss import MeanSquaredError
from dnn.test import check_gradients


def test_mean_squared_error():
    shape = (2, 3)
    input_data = [np.random.rand(*shape), np.random.rand(*shape)]

    op = MeanSquaredError()
    theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian), \
            "theorical: {}, numerical: {}".format(theorical_jacobian, numerical_jacobian)


if __name__ == '__main__':
    test_mean_squared_error()
