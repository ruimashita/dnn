# -*- coding: utf-8 -*-
from logging import getLogger

import numpy as np

from dnn.tensor import Tensor

logger = getLogger("dnn.operation")


class OperationMeta(type):
    """Operation Meta Class.

    Check forward() and backward() parameters by wrapping the methods.
    """

    @staticmethod
    def check_forward(func):
        """Check forward() parameters."""
        def func_wrapper(self, *inputs):
            logger.debug("forward func wrapper: %s %s %s", self, func, inputs)
            return func(self, *inputs)
        return func_wrapper

    @staticmethod
    def check_backward(func):
        """Check backward() parameters"""
        def func_wrapper(self, grad_outputs):
            if isinstance(grad_outputs, np.ndarray):
                assert isinstance(self.outputs, np.ndarray)
                assert self.outputs.shape == grad_outputs.shape

            elif isinstance(grad_outputs, tuple):
                assert len(grad_outputs) == len(self.outputs)

                for output, grad_output in zip(self.outputs, grad_outputs):
                    assert output.shape == grad_output.shape

            else:
                msg = "type error grad_outputs: {}".format(grad_outputs)
                raise TypeError(msg)

            grad_inputs = func(self, grad_outputs)
            logger.debug(
                "backwward func wrapper: %s %s %s %s %s %s",
                self, func, grad_outputs, self.outputs, self.inputs, grad_inputs
            )

            if isinstance(grad_inputs, np.ndarray):
                assert isinstance(grad_inputs, np.ndarray)
                assert grad_inputs.shape == self.inputs.shape

            elif isinstance(grad_inputs, tuple):
                assert len(self.inputs) == len(grad_inputs)

                for _input, grad_input in zip(self.inputs, grad_inputs):
                    assert _input.shape == grad_input.shape

            else:
                msg = "type error grad_inputs: {}".format(type(grad_inputs))
                raise TypeError(msg)

            return grad_inputs

        return func_wrapper

    @staticmethod
    def cast_from_tensor_to_data(func):
        """Cast function parameters from tensor to data."""
        def cast_wrapper(self, args):
            if isinstance(args, (tuple, list)):
                data_args = tuple([i.data for i in args])

            else:
                if isinstance(args, (Tensor)):
                    data_args = args.data

                else:
                    data_args = args

            returns = func(self, data_args)

            return returns

        return cast_wrapper

    def __new__(cls, cls_name, cls_bases, cls_dict):
        # wrap forward()
        if "forward" not in cls_dict:
            logger.error("no forward")
        forward_func = cls_dict["forward"]
        forward_func = OperationMeta.check_forward(forward_func)
        cls_dict["forward"] = forward_func

        # wrap backward()
        if "backward" not in cls_dict:
            logger.error("no backward")
        backward_func = cls_dict["backward"]
        backward_func = OperationMeta.check_backward(backward_func)
        backward_func = OperationMeta.cast_from_tensor_to_data(backward_func)
        cls_dict["backward"] = backward_func

        result = super().__new__(cls, cls_name, cls_bases, cls_dict)
        return result


class Operation(metaclass=OperationMeta):

    def __call__(self, *inputs):
        """Do the operation.

        Params:
        inputs: Tensors or a Tensor.

        Returns:
        outputs: python list of Tensors or a Tensor.
        """

        self.tensor_inputs = inputs
        self.inputs = tuple([
            _input.data if isinstance(_input, Tensor) else _input for _input in inputs
        ])

        outputs = self.forward(*self.inputs)

        logger.debug("outputs %s", outputs)

        if isinstance(outputs, np.ndarray):
            tensor_outputs = Tensor(outputs)
        elif isinstance(outputs, (tuple, list)):
            tensor_outputs = tuple([Tensor(output) for output in outputs])
        else:
            # outputs is scaler
            outputs = np.array([outputs])
            tensor_outputs = Tensor(outputs)

        if isinstance(tensor_outputs, Tensor):
            tensor_outputs.owner = self
        else:
            for tensor_output in tensor_outputs:
                tensor_output.owner = self

        # TODO(wakisaka): use weakref
        self.outputs = outputs

        return tensor_outputs

    def forward(self, *inputs):
        pass

    def backward(self, grad_outputs):
        """Backward.

        Params:
        grad_outputs: gradients of outputs. list of Tensor or a Tensor.
        """
        pass


def unbroadcast(grad, data):
    """Unbroadcast grad to data's shape.

    https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    """
    if grad.shape == data.shape:
        return grad

    dimention_broardcasted_axis = tuple(range(grad.ndim - data.ndim))
    grad = grad.sum(dimention_broardcasted_axis)

    size_one_broardcasted_axis = tuple(i for i, shape in enumerate(data.shape) if shape == 1)
    if len(size_one_broardcasted_axis) > 0:
        return grad.sum(keepdims=True, axis=size_one_broardcasted_axis)

    return grad


def needs_broadcast(*inputs):
    broadcasted = np.broadcast(*inputs)

    for _input in inputs:
        if _input.shape is not broadcasted.shape:
            return True

    return False
