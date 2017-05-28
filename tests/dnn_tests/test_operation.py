# -*- coding: utf-8 -*-
import numpy as np
import pytest

from dnn.operation import Operation
from dnn.tensor import Tensor


def test_not_implemented():

    with pytest.raises(NotImplementedError):
        class _NoForwardBackwardOperation(Operation):
            pass

    with pytest.raises(NotImplementedError):
        class _NoBackwardOperation(Operation):
            def forward(*inputs):
                return 1


class _ForwardReturnNotNdarrayOperation(Operation):
    def forward(*inputs):
        return 1, 2

    def backward(grad_outputs):
        return grad_outputs


def test_forward_return_not_ndarray():
    op = _ForwardReturnNotNdarrayOperation()

    with pytest.raises(AssertionError):
        op(1)

    with pytest.raises(AssertionError):
        op(np.array([1]))

    with pytest.raises(AssertionError):
        op(Tensor(np.array([1])))


class _BackwardOutputWrongLengthOperation(Operation):
    def forward(*inputs):
        return np.array([1]), np.array([2])

    def backward(grad_outputs):
        return grad_outputs


def test_backward_ouput_wrong_length():
    op = _BackwardOutputWrongLengthOperation()
    op(Tensor(np.array([1])))

    with pytest.raises(AssertionError):
        op.backward(Tensor(np.array([1])))


if __name__ == '__main__':
    test_not_implemented()
