# -*- coding: utf-8 -*-
from logging import getLogger

import numpy as np

from dnn.operation import Operation

logger = getLogger("dnn.operations.convolution")


def col2im(col, input_h, input_w, stride_y, stride_x, pad_h, pad_w):
    batch, input_c, kernel_h, kernel_w, out_h, out_w = col.shape

    img = np.zeros((batch,
                    input_c,
                    input_h + 2 * pad_h + stride_y - 1,
                    input_w + 2 * pad_w + stride_x - 1,
                    ),
                   dtype=col.dtype)

    for i in range(kernel_h):
        i_end = i + stride_y * out_h
        for j in range(kernel_w):
            j_end = j + stride_x * out_w

            img[:, :, i:i_end:stride_y, j:j_end:stride_x] += col[:, :, i, j, :, :]

    img = img[:, :, pad_h:input_h + pad_h, pad_w:input_w + pad_w]
    return img


def im2col(x, kernel_h, kernel_w, stride_y, stride_x, pad_h, pad_w):
    batch = x.shape[0]
    input_c = x.shape[1]
    input_h = x.shape[2]
    input_w = x.shape[3]
    out_h = 1 + ((input_h + 2 * pad_h - kernel_h) // stride_y)
    out_w = 1 + ((input_w + 2 * pad_w - kernel_w) // stride_x)

    # zero padding
    img = np.pad(
        x,
        ((0, 0), (0, 0), (pad_h, pad_h + stride_y - 1), (pad_w, pad_w + stride_x - 1)),
        mode='constant',
        constant_values=(0,))

    col = np.ndarray((batch, input_c, kernel_h, kernel_w, out_h, out_w), dtype=img.dtype)

    for i in range(kernel_h):
        i_end = i + stride_y * out_h
        for j in range(kernel_w):
            j_end = j + stride_x * out_w

            col[:, :, i, j, :, :] = img[:, :, i:i_end:stride_y, j:j_end:stride_x]

    return col


class Convolution2DBase():

    def __init__(self, stride=1, pad=0):

        if type(stride) is int:
            self.stride_y = stride
            self.stride_x = stride
        elif len(stride) == 2:
            self.stride_y, self.stride_x = stride
        else:
            raise Exception("Expected int, tuple/list with 2 entries. Got %s." % (type(stride)))

        if type(pad) is int:
            self.pad_h = pad
            self.pad_w = pad
        elif len(pad) == 2:
            self.pad_h, self.pad_w = pad
        else:
            raise Exception("Expected int, tuple/list with 2 entries. Got %s." % (type(pad)))


class Convolution2DIM2COL(Convolution2DBase, Operation):

    def forward(self, x, kernel):
        """

        x.shape -> batch, input_channel, height, width.
        kernel.shape -> output_channel, input_channel, kernel_height, kernel_width.
        """
        kernel_h = kernel.shape[2]
        kernel_w = kernel.shape[3]

        # col.shape -> batch, input_c, kernel_h, kernel_w, out_h, out_w
        col = im2col(x, kernel_h, kernel_w, self.stride_y, self.stride_x, self.pad_h, self.pad_w)

        # y.shape -> batch, out_h, out_w, out_c
        y = np.tensordot(
            col, kernel, ((1, 2, 3), (1, 2, 3)))

        y = np.moveaxis(y, 3, 1)
        return y

    def backward(self, grad_outputs):
        x, kernel = self.inputs

        kernel_h = kernel.shape[2]
        kernel_w = kernel.shape[3]
        col = im2col(x, kernel_h, kernel_w, self.stride_y, self.stride_x, self.pad_h, self.pad_w)

        # (b, out_c, out_h, out_w) * (b, in_c, k_h, k_w, out_h, out_w) -> out_c, in_c, k_h, k_w
        grad_kernel = np.tensordot(
            grad_outputs, col, ((0, 2, 3), (0, 4, 5)))

        # (out_c, in_c, k_h, k_w) * (b, out_c, h, w) -> (in_c, k_h, k_w, b, h, w)
        grad_col = np.tensordot(kernel, grad_outputs, (0, 1))
        grad_col = np.moveaxis(grad_col, 3, 0)  # (b, in_c, k_h, k_w, h, w)

        input_h = x.shape[2]
        input_w = x.shape[3]
        grad_x = col2im(grad_col, input_h, input_w, self.stride_y, self.stride_x, self.pad_h, self.pad_w)

        return grad_x, grad_kernel


class Convolution2DNaive(Convolution2DBase, Operation):

    def forward(self, x, kernel):
        """

        x.shape -> batch, input_channel, height, width.
        kernel.shape -> output_channel, input_channel, kernel_height, kernel_width.
        """
        batch = x.shape[0]
        input_h = x.shape[2]
        input_w = x.shape[3]
        out_c = kernel.shape[0]
        kernel_h = kernel.shape[2]
        kernel_w = kernel.shape[3]
        out_h = 1 + ((input_h + 2 * self.pad_h - kernel_h) // self.stride_y)
        out_w = 1 + ((input_w + 2 * self.pad_w - kernel_w) // self.stride_x)

        y = np.empty((batch, out_c, out_h, out_w))

        # zero padding
        x = np.pad(
            x,
            ((0, 0),
             (0, 0),
             (self.pad_h, self.pad_h + self.stride_y - 1),
             (self.pad_w, self.pad_w + self.stride_x - 1)),
            mode='constant',
            constant_values=(0,))

        for h in range(out_h):
            for w in range(out_w):
                new_h = h * self.stride_y
                new_w = w * self.stride_x
                # partial_input.shape -> b, in_c, k_h, k_w
                partial_input = x[:, :, new_h:new_h+kernel_h, new_w:new_w+kernel_w]
                tmp = np.tensordot(partial_input, kernel, ((1, 2, 3), (1, 2, 3)))
                y[:, :, h, w] = tmp

        return y

    def backward(self, grad_outputs):
        x, kernel = self.inputs

        input_h = x.shape[2]
        input_w = x.shape[3]
        kernel_h = kernel.shape[2]
        kernel_w = kernel.shape[3]
        out_h = 1 + ((input_h + 2 * self.pad_h - kernel_h) // self.stride_y)
        out_w = 1 + ((input_w + 2 * self.pad_w - kernel_w) // self.stride_x)

        # zero padding
        x = np.pad(
            x,
            ((0, 0),
             (0, 0),
             (self.pad_h, self.pad_h + self.stride_y - 1),
             (self.pad_w, self.pad_w + self.stride_x - 1)),
            mode='constant',
            constant_values=(0,))

        grad_x = np.zeros(x.shape)
        grad_kernel = np.zeros(kernel.shape)

        for h in range(out_h):
            for w in range(out_w):
                new_h = h * self.stride_y
                new_w = w * self.stride_x
                # partial_input.shape -> b, in_c, k_h, k_w
                partial_input = x[:, :, new_h:new_h+kernel_h, new_w:new_w+kernel_w]
                # partial_output.shape -> b, out_c
                partial_output = grad_outputs[:, :, h, w]
                tmp = np.tensordot(partial_output, partial_input, (0, 0))
                grad_kernel[:, :, :, :] += tmp

                tmp = np.tensordot(partial_output, kernel, ((1), (0)))
                grad_x[:, :, new_h:new_h+kernel_h, new_w:new_w+kernel_w] += tmp

        grad_x = grad_x[:, :, self.pad_h:self.pad_h+input_h, self.pad_w:self.pad_h+input_w]

        return grad_x, grad_kernel


class Convolution2DKN2ROW(Convolution2DBase, Operation):

    def forward(self, x, kernel):
        """

        x.shape -> batch, input_channel, height, width.
        kernel.shape -> output_channel, input_channel, kernel_height, kernel_width.
        """
        batch = x.shape[0]
        input_h = x.shape[2]
        input_w = x.shape[3]
        out_c = kernel.shape[0]
        kernel_h = kernel.shape[2]
        kernel_w = kernel.shape[3]
        out_h = 1 + ((input_h + 2 * self.pad_h - kernel_h) // self.stride_y)
        out_w = 1 + ((input_w + 2 * self.pad_w - kernel_w) // self.stride_x)

        # zero padding
        x = np.pad(
            x,
            ((0, 0),
             (0, 0),
             (self.pad_h, self.pad_h + self.stride_y - 1),
             (self.pad_w, self.pad_w + self.stride_x - 1)),
            mode='constant',
            constant_values=(0,))

        # 1x1 convolution
        # tmp.shape -> (batch, input_height+pad, input_widht+pad, out_channel, kernel_h, kernel_w)
        tmp = np.tensordot(
            x, kernel, ((1, ), (1, )))

        out = np.zeros((batch, out_c, out_h, out_w))
        for h in range(out_h):
            for w in range(out_w):
                stride_h = h * self.stride_y
                stride_w = w * self.stride_x
                for k_h in range(kernel_h):
                    for k_w in range(kernel_w):
                        out[:, :, h, w] += tmp[:, stride_h+k_h, stride_w+k_w, :, k_h, k_w]

        return out

    def backward(self, grad_outputs):
        pass


def convolution2d(x, stride=1, pad=0):
    return Convolution2DNaive(stride=stride, pad=pad)(x)
