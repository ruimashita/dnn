# -*- coding: utf-8 -*-
import math

import numpy as np

from dnn.gradient import gradients, get_variables


class Optimizer:

    def minimize(self, loss, tensors=None):
        self.loss = loss
        if tensors is None:
            tensors = list(get_variables(loss))
        self.tensors = tensors

        return self

    def update(self, loss_func=None, *args, **kwargs):
        if loss_func is None:
            loss = self.loss
        else:
            loss = loss_func(*args, **kwargs)
        grads = gradients(loss, self.tensors)

        for grad, tensor in zip(grads, self.tensors):
            self.step(tensor, grad)

    def step(self, tensor, grad, state=None):
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, tensor, grad, state=None):
        tensor.data = tensor.data - self.learning_rate * grad


class Adam(Optimizer):

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.global_step = 1

    def _setup(self):
        self.m_dict = {}
        self.v_dict = {}

        for tensor in self.tensors:
            # TODO(wakisaka): numpy
            self.m_dict[id(tensor)] = np.zeros_like(tensor.data)
            self.v_dict[id(tensor)] = np.zeros_like(tensor.data)

    def update(self, loss_func=None, *args, **kwargs):
        if self.global_step == 1:
            self._setup()
        super().update(loss_func, *args, **kwargs)
        self.global_step += 1

    def step(self, tensor, grad, state=None):
        m = self.m_dict[id(tensor)]
        v = self.v_dict[id(tensor)]

        # update m and v.
        m += (1. - self.beta1) * (grad - m)
        v += (1. - self.beta2) * (grad * grad - v)

        learning_rate = self.learning_rate \
            * math.sqrt(1. - self.beta2 ** self.global_step) \
            / (1. - self.beta1 ** self.global_step)

        # TODO(wakisaka): numpy
        tensor.data = tensor.data - learning_rate * m / (np.sqrt(v) + self.eps)
