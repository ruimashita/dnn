# -*- coding: utf-8 -*-
from logging import getLogger

import numpy as np

from dnn.operation import Operation

logger = getLogger("dnn.operations.activation")


class Relu(Operation):
    def forward(self, x):
        # TODO(wakisaka): numpy
        return np.maximum(x, 0.0, dtype=x.dtype)

    def backward(self, grad_outputs):
        x = self.inputs[0]
        return (x > 0.0) * grad_outputs,


def relu(x):
    return Relu()(x)


class Softmax(Operation):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        # TODO(wakisaka): numpy
        shift_x = x - x.max(axis=self.axis, keepdims=True)
        exps = np.exp(shift_x)
        return exps / exps.sum(axis=self.axis, keepdims=True)

    def backward(self, grad_outputs):
        output = self.outputs
        grad = output * (grad_outputs - (grad_outputs * output).sum(axis=self.axis, keepdims=True))
        return grad,


def softmax(a, axis=1):
    return Softmax(axis=1)(a)
