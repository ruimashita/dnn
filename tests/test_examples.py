# -*- coding: utf-8 -*-
import math
import unittest

import numpy as np
import tensorflow as tf

from examples.mnist_softmax import mnist_data, mnist_softmax
from examples.iris_mlp import iris_data, mlp_training


class TestMnistSoftmax(unittest.TestCase):

    @staticmethod
    def tensorflow_mnist_softmax():
        train_mnist_batch = mnist_data("train")

        graph = tf.Graph()

        with graph.as_default():
            images = tf.placeholder(tf.float32, [None, 784])
            labels = tf.placeholder(tf.float32, [None, 10])

            weights = tf.Variable(tf.zeros([784, 10]))
            biases = tf.Variable(tf.zeros([10]))
            output = tf.nn.softmax(tf.matmul(images, weights) + biases)
            cross_entropy = tf.reduce_mean(
                -tf.reduce_sum(labels * tf.log(output), reduction_indices=[1])
            )
            train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            init = tf.global_variables_initializer()

        sess = tf.Session(graph=graph)
        sess.run(init)
        for step in range(100):
            images_data, labels_data = next(train_mnist_batch(batch_size=200))
            _, accuracy_value = sess.run([train_step, accuracy], feed_dict={images: images_data, labels: labels_data})
            print("step: {}, train accuracy: {}".format(step, accuracy_value))
        return accuracy_value

    def test_accuracy(self):
        expect = TestMnistSoftmax.tensorflow_mnist_softmax()
        result = mnist_softmax()

        assert expect == result


class TestIrisMlp(unittest.TestCase):

    @staticmethod
    def tesnsorflow(train_data, test_data, train_labels, test_labels):
        np.random.seed(seed=32)
        graph = tf.Graph()

        with graph.as_default():
            data = tf.placeholder(tf.float32, [None, 4])
            labels = tf.placeholder(tf.float32, [None, 3])

            with tf.variable_scope("layer1"):
                num_first = 10
                num_input = 4
                num_output = num_first
                first_weights = tf.Variable(initial_value=np.random.normal(size=(num_input, num_output)),
                                            dtype=tf.float32)
                first_bias = tf.Variable(tf.zeros([num_output]), dtype=tf.float32)
                output = tf.matmul(data, first_weights) + first_bias
                output = tf.nn.relu(output)

            with tf.variable_scope("layer2"):
                num_input = num_first
                num_output = 3
                last_weights = tf.Variable(initial_value=np.random.normal(size=(num_input, num_output)),
                                           dtype=tf.float32)
                last_bias = tf.Variable(tf.zeros([num_output]), dtype=tf.float32)
                output = tf.matmul(output, last_weights) + last_bias

            output = tf.nn.softmax(output)
            loss = tf.reduce_mean(
                -tf.reduce_sum(labels * tf.log(output), reduction_indices=[1])
            )
            optimizer = tf.train.AdamOptimizer(0.01)
            train_op = optimizer.minimize(loss)

            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            init_op = tf.global_variables_initializer()

        sess = tf.Session(graph=graph)
        sess.run(init_op)
        for i in range(100):
            _, loss_value, accuracy_value = sess.run(
                [train_op, loss, accuracy],
                feed_dict={data: train_data, labels: train_labels}
            )
            yield loss_value, accuracy_value

        loss_value, accuracy_value = sess.run([loss, accuracy], feed_dict={data: test_data, labels: test_labels})
        yield loss_value, accuracy_value

    def test_loss_accuracy(self):
        train_data, test_data, train_labels, test_labels = iris_data()
        results = mlp_training(train_data, test_data, train_labels, test_labels)

        tensorflow_results = TestIrisMlp.tesnsorflow(train_data, test_data, train_labels, test_labels)

        for i, (tensorflow_result, result) in enumerate(zip(tensorflow_results, results)):
            tf_loss, tf_accuracy = tensorflow_result
            loss, accuracy = result
            print(tf_accuracy)
            assert math.isclose(tf_loss, loss, rel_tol=1e-05), "step {}".format(i)
            assert tf_accuracy == accuracy, "step {}".format(i)


if __name__ == "__main__":
    unittest.main()
