# -*- coding: utf-8 -*-
import numpy as np
import pytest

from dnn.tensor import Tensor
from dnn.operations.math import Add, Matmul, Multiply, Sum, Truediv, Pow, Subtract
from dnn.test import check_gradients


@pytest.fixture(params=[(4, 2, 3), (4, 2, 3), (2, 3), (2, 3)])
def a_shape(request):
    return request.param


@pytest.fixture(params=[(2, 1,), (3,), (2, 3), (1,)])
def b_shape(request):
    return request.param


def test_add():
    a_shape = (2, 3)
    b_shape = (3,)
    a = Tensor(np.random.rand(*a_shape))
    b = Tensor(np.random.rand(*b_shape))

    op = Add()

    input_data = a.data, b.data
    theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian), \
            "theorical: {}, numerical: {}".format(theorical_jacobian, numerical_jacobian)


def test_matmul():
    a_shape = (2, 3)
    b_shape = (3, 6)
    a = Tensor(np.random.rand(*a_shape))
    b = Tensor(np.random.rand(*b_shape))

    op = Matmul()

    input_data = a.data, b.data
    theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian), \
            "theorical: {}, numerical: {}".format(theorical_jacobian, numerical_jacobian)


def test_multiply():
    a_shape = (2, 3)
    b_shape = (2, 3)
    a = Tensor(np.random.rand(*a_shape))
    b = Tensor(np.random.rand(*b_shape))

    op = Multiply()

    forward_expects = a.data * b.data
    forward_results = op(a, b)

    assert np.allclose(forward_expects, forward_results.data)

    input_data = a.data, b.data
    theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian), \
            "theorical: {}, numerical: {}".format(theorical_jacobian, numerical_jacobian)


def test_multiply_broadcast(a_shape, b_shape):
    a_data = np.random.rand(*a_shape)
    b_data = np.random.rand(*b_shape)
    a = Tensor(a_data)
    b = Tensor(b_data)

    out_shape = np.broadcast(a_data, b_data).shape

    op = Multiply()

    forward_expects = a_data * b_data
    forward_results = op(a, b)

    assert out_shape == forward_results.shape
    assert np.allclose(forward_expects, forward_results.data)

    input_data = a.data, b.data
    theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian), \
            "theorical: {}, numerical: {}".format(theorical_jacobian, numerical_jacobian)


def test_pow():
    a_shape = (2, 3)
    b_shape = (2, 3)
    a_data = np.random.rand(*a_shape)
    b_data = np.random.rand(*b_shape)
    a = Tensor(a_data)
    b = Tensor(b_data)

    op = Pow()

    forward_expects = a_data ** b_data
    forward_results = op(a, b)

    assert np.allclose(forward_expects, forward_results.data)

    input_data = a_data, b_data
    theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian), \
            "theorical: {}, numerical: {}".format(theorical_jacobian, numerical_jacobian)


def test_pow_broadcast(a_shape, b_shape):
    a_data = np.random.rand(*a_shape)
    b_data = np.random.rand(*b_shape)
    a = Tensor(a_data)
    b = Tensor(b_data)

    out_shape = np.broadcast(a_data, b_data).shape

    op = Pow()

    forward_expects = a_data ** b_data
    forward_results = op(a, b)

    assert out_shape == forward_results.shape
    assert np.allclose(forward_expects, forward_results.data)

    input_data = a_data, b_data
    theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian), \
            "theorical: {}, numerical: {}".format(theorical_jacobian, numerical_jacobian)


def test_subtract():
    a_shape = (2, 3)
    b_shape = (3,)
    a = Tensor(np.random.rand(*a_shape))
    b = Tensor(np.random.rand(*b_shape))

    op = Subtract()

    input_data = a.data, b.data
    theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian), \
            "theorical: {}, numerical: {}".format(theorical_jacobian, numerical_jacobian)


def test_sum_axis():
    a_shape = (2, 3)
    a = Tensor(np.random.rand(*a_shape))

    out_shape = (2,)

    op = Sum(axis=1)
    forward_results = op(a)

    assert forward_results.data.shape == out_shape

    input_data = a.data
    theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian), \
            "theorical: {}, numerical: {}".format(theorical_jacobian, numerical_jacobian)


def test_sum():
    a_shape = (6,)
    a = Tensor(np.random.rand(*a_shape))

    out_shape = (1,)

    op = Sum(axis=None)
    forward_results = op(a)

    assert forward_results.data.shape == out_shape

    input_data = a.data
    theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian), \
            "theorical: {}, numerical: {}".format(theorical_jacobian, numerical_jacobian)


def test_truediv():
    a_shape = (2, 3)
    b_shape = (2, 3)
    a = Tensor(np.random.rand(*a_shape))
    b = Tensor(np.random.rand(*b_shape))

    op = Truediv()

    input_data = a.data, b.data
    theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian), \
            "theorical: {}, numerical: {}".format(theorical_jacobian, numerical_jacobian)


def test_truediv_broadcast(a_shape, b_shape):
    a_data = np.random.rand(*a_shape)
    b_data = np.random.rand(*b_shape)
    a = Tensor(a_data)
    b = Tensor(b_data)

    out_shape = np.broadcast(a_data, b_data).shape

    op = Truediv()

    forward_expects = a_data / b_data
    forward_results = op(a, b)

    assert out_shape == forward_results.shape
    assert np.allclose(forward_expects, forward_results.data)

    input_data = a.data, b.data
    theorical_jacobians, numerical_jacobians = check_gradients(op, input_data)
    for theorical_jacobian, numerical_jacobian in zip(theorical_jacobians, numerical_jacobians):
        assert np.allclose(theorical_jacobian, numerical_jacobian), \
            "theorical: {}, numerical: {}".format(theorical_jacobian, numerical_jacobian)


if __name__ == '__main__':
    test_add()
    test_matmul()
    test_multiply()
    test_pow()
    test_subtract()
    test_sum()
    test_sum_axis()
    test_truediv()
