# -*- coding: utf-8 -*-
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from dnn.tensor import Tensor, Variable
from dnn import operations as ops
from dnn.optimizer import Adam


def iris_data():
    iris = load_iris()
    data = iris.data

    data_std = np.copy(data)
    data_std[:, 0] = (data[:, 0] - data[:, 0].mean()) / data[:, 0].std()
    data_std[:, 1] = (data[:, 1] - data[:, 1].mean()) / data[:, 1].std()
    data_std[:, 2] = (data[:, 2] - data[:, 2].mean()) / data[:, 2].std()
    data_std[:, 3] = (data[:, 3] - data[:, 3].mean()) / data[:, 3].std()

    labels = iris.target
    labels = LabelBinarizer().fit_transform(labels)

    data = data_std

    train_data, test_data, train_labels, test_labels = train_test_split(
        data,
        labels,
        random_state=1,
        test_size=0.1,
    )

    return train_data, test_data, train_labels, test_labels


def mlp():
    num_first = 10
    num_input = 4
    num_output = num_first

    first_weights = Variable(np.random.normal(size=(num_input, num_output)))
    first_bias = Variable(np.zeros(num_output))

    num_input = num_output
    num_output = 3

    last_weights = Variable(np.random.normal(size=(num_input, num_output)))
    last_bias = Variable(np.zeros(num_output))

    optimizer = Adam(0.01)

    def call(data, labels, is_training=False):
        first_output = data @ first_weights + first_bias
        first_output = ops.relu(first_output)

        last_output = first_output @ last_weights + last_bias

        output = ops.softmax(last_output)

        cross_entropy = - ops.sum(labels * ops.log(output), axis=1)
        loss = ops.mean(cross_entropy)

        if is_training:
            optimize = optimizer.minimize(loss)
            optimize.update()

        correct_prediction = np.equal(np.argmax(output.data, 1), np.argmax(labels.data, 1))
        accuracy = np.mean(correct_prediction, dtype=np.float32)

        return np.float32(loss.data)[0], accuracy

    return call


def mlp_training(train_data, test_data, train_labels, test_labels):
    np.random.seed(seed=32)
    train_data = Tensor(train_data, name="train_datas")
    train_labels = Tensor(train_labels, name="train_labels")

    test_data = Tensor(test_data, name="test_datas")
    test_labels = Tensor(test_labels, name="test_labels")

    model = mlp()
    for i in range(100):
        loss, accuracy = model(train_data, train_labels, is_training=True)
        yield loss, accuracy

    loss, accuracy = model(test_data, test_labels, is_training=False)
    yield loss, accuracy


if __name__ == '__main__':
    train_data, test_data, train_labels, test_labels = iris_data()
    results = mlp_training(train_data, test_data, train_labels, test_labels)

    for i, (loss, accuracy) in enumerate(results):
        print("step: {}, loss {}, accuracy {}".format(i, loss, accuracy))
