# -*- coding: utf-8 -*-
from logging import getLogger

import numpy as np

from dnn.tensor import Tensor
from dnn.operation import Operation, unbroadcast, needs_broadcast

logger = getLogger("dnn.operations.math")


class Add(Operation):

    def forward(self, a, b):
        output = a + b
        return output

    def backward(self, grad_outputs):
        a, b = self.inputs

        grad_a = 1 * grad_outputs
        grad_b = 1 * grad_outputs

        if needs_broadcast(*self.inputs):
            grad_a = unbroadcast(grad_a, a)
            grad_b = unbroadcast(grad_b, b)

        return grad_a, grad_b


def add(a, b):
    if isinstance(b, (int, float)):
        b = Tensor(np.array([b]))

    result = Add()(a, b)
    result.name = "{} + {}".format(a.name, b.name)
    return result


class Log(Operation):

    def forward(self, a):
        # TODO(wakisaka): numpy
        output = np.log(a)
        return output

    def backward(self, grad_outputs):
        a, = self.inputs
        grad_a = 1/a * grad_outputs
        return grad_a,


def log(a):
    return Log()(a)


class Matmul(Operation):

    def forward(self, a, b):
        output = a @ b
        return output

    def backward(self, grad_outputs):
        a, b = self.inputs
        grad_a = grad_outputs @ b.T
        grad_b = a.T @ grad_outputs
        return grad_a, grad_b


def matmul(a, b):
    return Matmul()(a, b)


class Multiply(Operation):

    def forward(self, a, b):
        output = a * b
        return output

    def backward(self, grad_outputs):
        a, b = self.inputs
        grad_a = b * grad_outputs
        grad_b = a * grad_outputs

        if needs_broadcast(*self.inputs):
            grad_a = unbroadcast(grad_a, a)
            grad_b = unbroadcast(grad_b, b)
        return grad_a, grad_b


def multiply(a, b):
    if isinstance(b, (int, float)):
        b = Tensor(np.array([b]))
    result = Multiply()(a, b)
    result.name = "{} * {}".format(a.name, b.name)
    return result


class Pow(Operation):

    def forward(self, a, b):
        output = a ** b
        return output

    def backward(self, grad_outputs):
        a, b = self.inputs

        grad_a = (b * (a ** (b - 1.))) * grad_outputs
        # TODO(wakisaka): numpy
        grad_b = (np.log(a) * self.outputs) * grad_outputs

        if needs_broadcast(*self.inputs):
            grad_a = unbroadcast(grad_a, a)
            grad_b = unbroadcast(grad_b, b)

        return grad_a, grad_b


class Subtract(Operation):

    def forward(self, a, b):
        output = a - b
        return output

    def backward(self, grad_outputs):
        a, b = self.inputs

        grad_a = 1 * grad_outputs
        grad_b = -1 * grad_outputs

        if needs_broadcast(*self.inputs):
            grad_a = unbroadcast(grad_a, a)
            grad_b = unbroadcast(grad_b, b)

        return grad_a, grad_b


class Sum(Operation):
    def __init__(self, axis=None):
        self.axis = axis

    def forward(self, a):
        return a.sum(axis=self.axis)

    def backward(self, grad_outputs):
        a = self.inputs[0]
        if self.axis is not None:
            # TODO(wakisaka): numpy
            grad_outputs = np.expand_dims(grad_outputs, axis=self.axis)
        _, grad = np.broadcast_arrays(a, grad_outputs)
        return grad,


def sum(a, axis=None):
    return Sum(axis=axis)(a)


class Truediv(Operation):

    def forward(self, a, b):
        output = a / b
        return output

    def backward(self, grad_outputs):
        a, b = self.inputs
        grad_a = 1/b * grad_outputs
        grad_b = -(a / (b ** 2)) * grad_outputs

        if needs_broadcast(*self.inputs):
            grad_a = unbroadcast(grad_a, a)
            grad_b = unbroadcast(grad_b, b)

        return grad_a, grad_b


def mean(a, axis=None):
    if axis is None:
        divider = a.data.size
    else:
        divider = a.data.shape[axis]

    divider = Tensor(np.array([divider]))
    sumed = Sum(axis)(a)
    return sumed / divider
