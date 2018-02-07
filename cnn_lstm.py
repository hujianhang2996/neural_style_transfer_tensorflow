import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import BasicLSTMCell


seq_length = 48
X = np.random.rand(10, seq_length, 1)
Y = np.random.rand(10, 2, 1)

df_graph = tf.Graph()
with df_graph.as_default():
    batch_size = tf.placeholder(tf.int32, [])
    input_X = tf.placeholder(tf.float32, (None, seq_length, 1), 'input_X')
    input_Y = tf.placeholder(tf.float32, (None, 2, 1), 'input_Y')

    conv1_out = tf.layers.conv1d(input_X, 2, 13, activation=tf.nn.relu, name='conv1')
    conv2_out = tf.layers.conv1d(conv1_out, 2, 13, activation=tf.nn.relu, name='conv2')

    pooling_out = tf.layers.average_pooling1d(conv2_out, 2, 2, name='pooling')

    conv3_out = tf.layers.conv1d(pooling_out, 4, 5, activation=tf.nn.relu, name='conv3')
    conv4_out = tf.layers.conv1d(conv3_out, 4, 5, activation=tf.nn.relu, name='conv4')

    pooling1_out = tf.layers.average_pooling1d(conv4_out, 2, 2, name='pooling1')

    resort_out = tf.transpose(pooling1_out, [1, 0, 2], name='resort')

    lstm_layer = BasicLSTMCell(1)
    state = lstm_layer.zero_state(batch_size, tf.float32)

    out = []
    for i in range(2):
        output, state = lstm_layer(resort_out[i], state)
        out.append(output)

    out_gather = [conv4_out, pooling1_out, resort_out, out]
    init_op = tf.global_variables_initializer()

sess = tf.Session(graph=df_graph)
sess.run(init_op)
train_writer = tf.summary.FileWriter('./cnn_lstm', sess.graph, flush_secs=5)
out_run = sess.run(out_gather, feed_dict={input_X: X, input_Y: Y, batch_size: X.shape[0]})
[print(np.array(x).shape) for x in out_run]

