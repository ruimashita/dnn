# -*- coding: utf-8 -*-
import numpy as np

from logging import getLogger

logger = getLogger("dnn.tensor")


class Tensor(object):
    def __init__(self, data, name=None):
        if not isinstance(data, np.ndarray):
            msg = "numpy.ndarray are expected. Actual: {0}".format(type(data))
            raise TypeError(msg)

        self.data = data
        self.name = name
        self._owner = None

    def __str__(self):
        return "Tensor(name:{}, data.shape:{}, data:{},)".format(
            self.name, self.data.shape, self.data
        )

    def __repr__(self):
        return str(self)

    @property
    def shape(self):
        return self.data.shape

    @property
    def owner(self):
        """Creater `Operation` of the tensor"""
        return self._owner

    @owner.setter
    def owner(self, value):
        self._owner = value

    def __hash__(self):
        return hash(self.data.tostring())

    def __add__(self, other):
        from dnn.operations.math import add
        return add(self, other)

    def __sub__(self, other):
        from dnn.operations.math import Subtract
        result = Subtract()(self, other)
        result.name = "{} - {}".format(self.name, other.name)
        return result

    def __pow__(self, other):
        from dnn.operations.math import Pow
        if isinstance(other, int):
            other = Tensor(np.array([other]))
        result = Pow()(self, other)
        result.name = "{} ** {}".format(self.name, other.name)
        return result

    def __truediv__(self, other):
        from dnn.operations.math import Truediv
        result = Truediv()(self, other)
        result.name = "{} / {}".format(self.name, other.name)
        return result

    def __matmul__(self, other):
        from dnn.operations.math import Matmul
        result = Matmul()(self, other)
        result.name = "{} @ {}".format(self.name, other.name)
        return result

    def __mul__(self, other):
        from dnn.operations.math import multiply
        return multiply(self, other)

    def __neg__(self):
        return self * Tensor(np.array([-1]))


class Variable(Tensor):
    def __init__(self, data, name=None):
        super().__init__(data, name=name)
