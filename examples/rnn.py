#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import math

import tensorflow as tf
import numpy as np
import pandas as pd


def make_mini_batch(train_data, size_of_mini_batch, length_of_sequences):
    inputs = np.empty(0)
    outputs = np.empty(0)
    for _ in range(size_of_mini_batch):
        index = random.randint(0, len(train_data) - length_of_sequences)
        part = train_data[index:index + length_of_sequences]
        inputs = np.append(inputs, part[:, 0])
        outputs = np.append(outputs, part[-1, 1])
    inputs = inputs.reshape(-1, length_of_sequences, 1)
    outputs = outputs.reshape(-1, 1)
    return (inputs, outputs)


def make_prediction_initial(train_data, index, length_of_sequences):
    return train_data[index:index + length_of_sequences, 0]


train_data_path = "../train_data/normal.npy"
num_of_input_nodes = 1
num_of_hidden_nodes = 2
num_of_output_nodes = 1
length_of_sequences = 2
num_of_training_epochs = 2000
length_of_initial_sequences = 50
num_of_prediction_epochs = 100
size_of_mini_batch = 100
learning_rate = 0.1
forget_bias = 1.0
print("train_data_path             = %s" % train_data_path)
print("num_of_input_nodes          = %d" % num_of_input_nodes)
print("num_of_hidden_nodes         = %d" % num_of_hidden_nodes)
print("num_of_output_nodes         = %d" % num_of_output_nodes)
print("length_of_sequences         = %d" % length_of_sequences)
print("num_of_training_epochs      = %d" % num_of_training_epochs)
print("length_of_initial_sequences = %d" % length_of_initial_sequences)
print("num_of_prediction_epochs    = %d" % num_of_prediction_epochs)
print("size_of_mini_batch          = %d" % size_of_mini_batch)
print("learning_rate               = %f" % learning_rate)
print("forget_bias                 = %f" % forget_bias)


def generate_train_data():
    # サイクルあたりのステップ数
    steps_per_cycle = 50
    # 生成するサイクル数
    number_of_cycles = 100
    df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
    df["sin_t"] = df.t.apply(lambda x: math.sin(x * (2 * math.pi / steps_per_cycle)))
    df["sin_t+1"] = df["sin_t"].shift(-1)
    df.dropna(inplace=True)
    return df[["sin_t", "sin_t+1"]].as_matrix()

def main():
    train_data = generate_train_data()
    print("train_data:", train_data)

    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    with tf.Graph().as_default():
        input_ph = tf.placeholder(tf.float32, [None, length_of_sequences, num_of_input_nodes], name="input")
        supervisor_ph = tf.placeholder(tf.float32, [None, num_of_output_nodes], name="supervisor")
        istate_ph = tf.placeholder(tf.float32, [None, num_of_hidden_nodes * 2], name="istate")

        with tf.name_scope("inference") as scope:
            with tf.name_scope("inputs") as scope:
                weight1_var = tf.Variable(tf.truncated_normal(
                    [num_of_input_nodes, num_of_hidden_nodes], stddev=0.1), name="weight1")
                bias1_var = tf.Variable(tf.truncated_normal([num_of_hidden_nodes], stddev=0.1), name="bias1")
     
                # (batch, sequence, data) -> (sequence, batch, data)
                in1 = tf.transpose(input_ph, [1, 0, 2])

                # (sequence, batch, data) -> (sequence * batch, data)
                in2 = tf.reshape(in1, [-1, num_of_input_nodes])

                #  (sequence * batch, data) -> (sequence * batch, num_of_hidden_nodes)
                in3 = tf.matmul(in2, weight1_var) + bias1_var

                # sequence * (batch, num_of_hidden_nodes)
                in4 = tf.split(in3, length_of_sequences)

            cell = tf.contrib.rnn.BasicLSTMCell(num_of_hidden_nodes, forget_bias=forget_bias, state_is_tuple=False)
            # rnn_output, states_op = rnn.rnn(cell, in4, initial_state=istate_ph)
            rnn_output, states_op = tf.contrib.rnn.static_rnn(cell, in4, initial_state=istate_ph)
            with tf.name_scope("output") as scope:
                bias2_var = tf.Variable(tf.truncated_normal([num_of_output_nodes], stddev=0.1), name="bias2")
                weight2_var = tf.Variable(tf.truncated_normal(
                    [num_of_hidden_nodes, num_of_output_nodes], stddev=0.1), name="weight2")
                output_op = tf.matmul(rnn_output[-1], weight2_var) + bias2_var

        with tf.name_scope("loss") as scope:
            square_error = tf.reduce_mean(tf.square(output_op - supervisor_ph))
            loss_op = square_error
            tf.summary.scalar("loss", loss_op)

        with tf.name_scope("training") as scope:
            training_op = optimizer.minimize(loss_op)

        summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter("data", graph=sess.graph)
            sess.run(init)

            for epoch in range(num_of_training_epochs):
                inputs, supervisors = make_mini_batch(train_data, size_of_mini_batch, length_of_sequences)

                train_dict = {
                    input_ph:      inputs,
                    supervisor_ph: supervisors,
                    istate_ph:     np.zeros((size_of_mini_batch, num_of_hidden_nodes * 2)),
                }
                sess.run(training_op, feed_dict=train_dict)

                if (epoch + 1) % 10 == 0:
                    summary_str, train_loss = sess.run([summary_op, loss_op], feed_dict=train_dict)
                    summary_writer.add_summary(summary_str, epoch)
                    print("train#%d, train loss: %e" % (epoch + 1, train_loss))

            inputs = make_prediction_initial(train_data, 0, length_of_initial_sequences)
            outputs = np.empty(0)
            states = np.zeros((num_of_hidden_nodes * 2)),

            print("initial:", inputs)
            np.save("initial.npy", inputs)

            for epoch in range(num_of_prediction_epochs):
                pred_dict = {
                    input_ph:  inputs.reshape((1, length_of_sequences, 1)),
                    istate_ph: states,
                }
                output, states = sess.run([output_op, states_op], feed_dict=pred_dict)
                print("prediction#%d, output: %f" % (epoch + 1, output))

                inputs = np.delete(inputs, 0)
                inputs = np.append(inputs, output)
                outputs = np.append(outputs, output)

            print("outputs:", outputs)
            np.save("output.npy", outputs)

            saver.save(sess, "data/model")



main()            
