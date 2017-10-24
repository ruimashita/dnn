# -*- coding: utf-8 -*-
import os
import gzip
import struct
import urllib.request

import numpy as np

from dnn.tensor import Tensor, Variable
from dnn import operations as ops
from dnn.optimizer import Adam

_MNIST_URL = "http://yann.lecun.com/exdb/mnist/"

_TRAIN_IMAGE_FILE = "train-images-idx3-ubyte.gz"
_TRAIN_LABEL_FILE = "train-labels-idx1-ubyte.gz"
_TEST_IMAGE_FILE = "t10k-images-idx3-ubyte.gz"
_TEST_LABEL_FILE = "t10k-labels-idx1-ubyte.gz"

_DATA_DIR = os.path.join("data", "MNIST", "dir")


def _maybe_download_and_parse(filename):
    target_file = os.path.join(_DATA_DIR, filename)
    download_url = _MNIST_URL + filename

    if not os.path.exists(_DATA_DIR):
        os.makedirs(_DATA_DIR)

    if not os.path.exists(target_file):
        urllib.request.urlretrieve(download_url, target_file)

    with gzip.open(target_file) as f:
        if "images" in filename:
            header, num, rows, cols = struct.unpack(">4i", f.read(16))
            size = num * rows * cols
            data = struct.unpack("{}B".format(size), f.read(size))
            data = np.array(data, dtype=np.uint8).reshape(num, rows, cols)

        if "labels" in filename:
            header, num = struct.unpack(">2i", f.read(8))
            data = struct.unpack("{}B".format(num), f.read(num))
            data = np.array(data, dtype=np.uint8)

    return data


def mnist_data(mode="train"):
    print("load {} mnist...".format(mode))
    if mode == "train":
        images = _maybe_download_and_parse(_TRAIN_IMAGE_FILE)
        labels = _maybe_download_and_parse(_TRAIN_LABEL_FILE)
    else:
        images = _maybe_download_and_parse(_TEST_IMAGE_FILE)
        labels = _maybe_download_and_parse(_TEST_LABEL_FILE)

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
    np.random.seed(seed=615)
    train_mnist = mnist_data("train")
    train_mnist_batch = train_mnist(batch_size=200)

    weights = Variable(np.random.normal(size=(784, 10)))
    biases = Variable(np.random.normal(size=(10)))

    optimizer = Adam(0.01)
    for step in range(100):
        images_data, labels_data = next(train_mnist_batch)

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
