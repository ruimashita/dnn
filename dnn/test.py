# -*- coding: utf-8 -*-
import copy

import numpy as np

from dnn.tensor import Tensor


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


def check_gradients(y, xs, epsilon=1e-8):
    """Calculate theorical and numerical Jacobian list for each inputs (xs).

    2-d numpy array representing the Jacobian for dy/dx in xs. \
    Each has ``y size`` rows and ``x size`` columns.
    ``y size`` is the number of elements in operations output
    and ``x size`` is the number of elements in input::

        [
            [dy[0]/dx[0], dy[0]/dx[1], ... , dy[0],dx[x_size]],
            [dy[1]/dx[0], dy[1]/dx[1], ... , dy[1],dx[x_size]],
            ...,
            ...,
            [dy[y_size]/dx[0], dy[y_size]/dx[1], ... , dy[y_size],dx[x_size]],
        ].

    Args:
        y(Operation): Operation to be partial derivatived.
        xs(tuple or list of numpy.ndarray): Inputs of operation.

    Returns:
        tuple:
            theorical_jacobians(list): list of theorical jacobians for each inputs(xs).
            numerical_jacobians(list): list of numerical jacobians for each inputs(xs).
    """
    xs = _as_tuple(xs)
    xs_tensor = [Tensor(x) for x in xs]

    outputs = y(*xs_tensor)
    if isinstance(outputs, tuple):
        # TODO(wakisaka): Implement tuple outputs pattern. the pattern cannot (yet) gradient with multiple outputs.
        raise Exception("cannot check gradient with multiple outputs")
    y_shape = outputs.shape
    theorical_jacobians = _theorical_jacobians(y, xs, y_shape)
    numerical_jacobians = _numerical_jacobians(y, xs, y_shape, epsilon)

    return theorical_jacobians, numerical_jacobians


def _theorical_jacobians(y, xs, y_shape):
    """Calculate theorical jacobian for each xs.

    Args:
        y(Operation): Operation to be partial derivatived.
        input_data(tuple of numpy.ndarray): Inputs of operation.
        y_shape(tuple): shape of operation(y) output.

    Returns:
        list of jacobian.
    """
    output_size = np.prod(y_shape)
    jacobians = []
    for input_index, x in enumerate(xs):
        # allocate jacobian matrix.
        jacobian = np.empty((output_size, x.size))

        # partial derivatives of y[element_indecies] w.r.t x.
        for element_indecies in np.ndindex(y_shape):
            # Gradients of output, we set one element to be 1.0 and everything else to be 0 like 1 of k.
            grad_outputs = np.zeros(shape=y_shape)
            grad_outputs[element_indecies] = 1.0

            # backward with 1 of k give us one row of the jacobian.
            # [dy[element_indecies]/dx]
            grad_inputs = y.backward(grad_outputs)

            # converts element_indecie into the flattened.
            index = np.ravel_multi_index(element_indecies, y_shape)
            jacobian[index, :] = grad_inputs[input_index].ravel()

        jacobians.append(jacobian)
    return jacobians


def _numerical_jacobians(y, xs, y_shape, epsilon):
    """Calculate numerical jacobian for each xs.

    Args:
        y(Operation): Operation to be partial derivatived.
        input_data(tuple of numpy.ndarray): Inputs of operation.
        y_shape(tuple): shape of operation(y) output.

    Returns:
        list of jacobian.
    """
    output_size = np.prod(y_shape)
    jacobians = []
    for input_index, x in enumerate(xs):

        # allocate jacobian matrix.
        jacobian = np.empty((output_size, x.size))

        # calculate partial derivative of y w.r.t traget(x[element_indecies]).
        for element_indecies in np.ndindex(x.shape):
            target = x[element_indecies]

            increase_inputs = copy.deepcopy(xs)
            increase_inputs[input_index][element_indecies] = target + epsilon

            # result of increaseing traget inputs. y(target + epsilon)
            increase_result = y.forward(*increase_inputs)

            reduce_inputs = copy.deepcopy(xs)
            reduce_inputs[input_index][element_indecies] = target - epsilon

            # result of reducing traget inputs. y(target - epsilon)
            reduce_result = y.forward(*reduce_inputs)

            # TODO(wakisaka): make pattern of results is tuple. cant (yet) gradient  with multiple outputs
            # partial derivatives of y w.r.t traget [dy[i] / dtraget].
            # It is one colomun of jacobian.
            result = (increase_result - reduce_result) / (2 * epsilon)

            # converts element_indecie into the flattened.
            index = np.ravel_multi_index(element_indecies, x.shape)
            jacobian[:, index] = result.ravel()

        jacobians.append(jacobian)
    return jacobians
