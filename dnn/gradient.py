# -*- coding: utf-8 -*-
from collections import OrderedDict
import functools

import numpy as np

from dnn.tensor import Variable


class _BackwardCache:
    """Cache backward"""

    def __init__(self):
        self.memo = {}
        self.operations = []

    def __call__(self, func):
        """Wrap backward to memorize"""
        closure = getattr(func, "__closure__")

        # if func is already wrapped, return func.
        if closure and type(closure[-1].cell_contents) is type(self):
            return func

        @functools.wraps(func)
        def wrapper(grad_outputs):
            operation = func.__self__
            self.operations.append(operation)
            _hash = hash((operation, grad_outputs.tostring()))
            if _hash not in self.memo:
                self.memo[_hash] = func(grad_outputs)
            return self.memo[_hash]
        return wrapper

    def clean(self):
        """Remove memorize wrapper"""
        for operation in self.operations:
            closure = getattr(operation.backward, "__closure__")
            if closure and type(closure[-1].cell_contents) is type(self):
                original_backward = closure[-2].cell_contents
                operation.backward = original_backward
        self.operations = []

    def __del__(self):
        self.clean()


def gradients(y, xs):
    """Compute graindents.

    Return the list of partial derivatives y with respect to x in xs.
    The list of length is `len(xs)`.

    Args:
        y(Tensor): A tensor to be differentiated.
        xs(list of Tensor): list of Tensor to be used for differentiated.

    Returns:
        List of partial derivatives y with respect to x in xs.
    """
    grad_y = np.ones_like(y.data)
    grads = []

    cache = _BackwardCache()

    for i, x in enumerate(xs):
        operation_and_index_path = OrderedDict(_traverse(y, x))

        grad_outputs = grad_y
        for operation, index in operation_and_index_path.items():
            operation.backward = cache(operation.backward)

            grad_inputs = operation.backward(grad_outputs)
            grad_outputs = grad_inputs[index]

        grads.append(grad_outputs)

    del cache
    return grads


def get_variables(y):
    """Find all variables form y.

    Args:
        y(Tensor): Graph root tensor.
    """
    if type(y) is Variable:
        yield y
        return

    operation = y.owner
    # if y is leaf
    if operation is None:
        return

    inputs = operation.tensor_inputs
    for _input in inputs:
        yield from get_variables(_input)


def _traverse(y, x):
    """Depth first traversal form y to x.

    Generate operation and operation input index tuple.

    Params:
    y: Tensor
    x: Tensor
    """
    # print("_traverse", y, x)
    operation = y.owner
    # if y is leaf
    if operation is None:
        return

    inputs = operation.tensor_inputs
    inputs_id = [id(_input) for _input in inputs]

    if id(x) in inputs_id:
        index = inputs_id.index(id(x))
        yield operation, index
        return

    for index, tmp_tensor in enumerate(inputs):

        results = list(_traverse(tmp_tensor, x))
        if len(results) > 0:
            yield operation, index
            for result in results:
                yield result
            return
