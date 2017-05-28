# -*- coding: utf-8 -*-
import tempfile
import numpy as np
import os

import mnist as mnist_loader

from dnn.tensor import Tensor, Variable
from dnn import operations as ops
from dnn.optimizer import Adam


def mnist_data(mode="train"):
    print("load {} mnist...".format(mode))
    tempfile.tempdir = "./MNIST_data"
    if not os.path.exists(tempfile.tempdir):
        os.mkdir(tempfile.tempdir)
    if mode == "train":
        images = mnist_loader.train_images()
        labels = mnist_loader.train_labels()
    else:
        images = mnist_loader.test_images()
        labels = mnist_loader.test_labels()
    tempfile.tempdir = None
    print("done")

    assert len(images) == len(labels)
    num_data = len(images)

    images = images / 255.0
    images = images.reshape((-1, 784))

    # onehot.
    labels = np.eye(10)[labels]

    start_index = 0

    def batch(batch_size=num_data):
        nonlocal start_index
        while True:
            end_index = start_index + batch_size
            data_slice = slice(start_index, end_index)
            yield images[data_slice], labels[data_slice]
            if end_index < num_data:
                start_index = end_index
            else:
                start_index = 0
    return batch


def mnist_softmax():
    train_mnist_batch = mnist_data("train")

    weights = Variable(np.zeros([784, 10]))
    biases = Variable(np.zeros([10]))

    optimizer = Adam(0.001)
    for step in range(100):
        images_data, labels_data = next(train_mnist_batch(batch_size=200))

        images = Tensor(images_data, name="datas")
        labels = Tensor(labels_data, name="labels")

        logits = images @ weights + biases
        output = ops.softmax(logits, axis=1)

        cross_entropy = - ops.sum(labels * ops.log(output), axis=1)
        loss = ops.mean(cross_entropy)

        optimizer = optimizer.minimize(loss)
        optimizer.update()

        correct_prediction = np.equal(np.argmax(output.data, 1), np.argmax(labels.data, 1))
        accuracy = np.mean(correct_prediction, dtype=np.float32)
        print("step: {}, train accuracy: {}".format(step, accuracy))

    return accuracy


if __name__ == '__main__':
    mnist_softmax()
