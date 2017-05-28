# -*- coding: utf-8 -*-
from logging import getLogger

from dnn.operation import Operation

logger = getLogger("dnn.operations.loss")


class MeanSquaredError(Operation):

    def forward(self, a, b):
        return (a - b) ** 2

    def backward(self, grad_outputs):
        a, b = self.inputs
        grad_a = 2 * (a - b) * grad_outputs
        grad_b = 2 * (b - a) * grad_outputs
        return grad_a, grad_b


def mean_squared_error(a, b):
    return MeanSquaredError()(a, b)
