# -*- coding: utf-8 -*-

import numpy as np

from dnn.tensor import Tensor, Variable
from dnn import operations
from dnn.gradient import gradients, get_variables


def test_gradients():
    x = Tensor(np.array([1]))
    y = x + 2
    z = y * 3

    grads = gradients(z, [y, x])

    assert grads[0] == np.array([3])
    assert grads[1] == np.array([3])


def test_get_variables():
    inputs = np.random.rand(10, 5)
    labels = np.random.rand(10, 5)
    init_weights = np.random.rand(10, 5)
    init_bias = np.random.rand(5)

    weights = Variable(init_weights, name="weights")
    bias = Variable(init_bias, name="bias")

    inputs = Tensor(inputs, name="inputs")
    labels = Tensor(labels, name="labels")

    tmp = inputs * weights
    tmp.name = "tmp"

    y = tmp + bias
    y.name = "y"

    mse = (y - labels) ** 2
    loss_tmp = operations.sum(mse, axis=1)
    loss = operations.mean(loss_tmp)

    variables = get_variables(loss)

    assert [id(weights), id(bias)] == [id(v) for v in variables]


if __name__ == '__main__':
    test_gradients()
    test_get_variables()
