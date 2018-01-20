'''
cifar10 test
'''
import os
import random
from pathlib import Path
import tensorflow as tf
import numpy as np
import vgg16_inference
import cifar10_reader

DELDIR1 = 'train'
DELDIR2 = 'test'
if Path(DELDIR1).exists():
    DELDIR1 = os.listdir(DELDIR1)
    for f in DELDIR1:
        filePath = os.path.join(DELDIR1, f)
        os.remove(filePath)
    os.rmdir(DELDIR1)
if Path(DELDIR2).exists():
    DELDIR2 = os.listdir(DELDIR2)
    for f in DELDIR2:
        filePath = os.path.join(DELDIR2, f)
        os.remove(filePath)
    os.rmdir(DELDIR2)

layerName_list_vgg16 = iter([
    'block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1',
    'block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2',
    'block3_conv3', 'block3_pool', 'block4_conv1', 'block4_conv2',
    'block4_conv3', 'block4_pool', 'block5_conv1', 'block5_conv2',
    'block5_conv3', 'block5_pool'
])
parameters_vgg16 = []
default_initializer = tf.contrib.layers.xavier_initializer
LAYER1_NODES = 200
LAYER2_NODES = 200
LAYER3_NODES = 200
DROP_OUT_RATE = 1
LAMBDA = 0.01
BATCH_SIZE = 100
EPOCH = 100
LEARNING_RATE = 0.01

# 定义网络输入
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_ = tf.placeholder(tf.int32, [
    None,
])
keep_prob = tf.placeholder(tf.float32)

# 构建vgg16的卷积层
y_block5_pool, _, _ = vgg16_inference.inference(x, layerName_list_vgg16,
                                                parameters_vgg16)

# flatten操作
shp = y_block5_pool.get_shape()
flattened_shape = shp[1].value * shp[2].value * shp[3].value
resh1 = tf.reshape(y_block5_pool, [-1, flattened_shape], name="reshape")

# 构建fc层
layer1_weights = tf.get_variable(
    name='layer1_weights',
    shape=[flattened_shape, LAYER1_NODES],
    initializer=default_initializer(),
    trainable=True)
layer1_biases = tf.get_variable(
    name='layer1_biases',
    shape=[LAYER1_NODES],
    initializer=tf.constant_initializer(0.1),
    trainable=True)
y = tf.nn.relu_layer(resh1, layer1_weights, layer1_biases, name='layer1')
y_drop = tf.nn.dropout(y, keep_prob, name="layer1_drop")

layer2_weights = tf.get_variable(
    name='layer2_weights',
    shape=[LAYER1_NODES, LAYER2_NODES],
    initializer=default_initializer(),
    trainable=True)
layer2_biases = tf.get_variable(
    name='layer2_biases',
    shape=[LAYER2_NODES],
    initializer=tf.constant_initializer(0.1),
    trainable=True)
y = tf.nn.relu_layer(y_drop, layer2_weights, layer2_biases, name='layer2')
y_drop = tf.nn.dropout(y, keep_prob, name="layer2_drop")

layer3_weights = tf.get_variable(
    name='layer3_weights',
    shape=[LAYER2_NODES, LAYER3_NODES],
    initializer=default_initializer(),
    trainable=True)
layer3_biases = tf.get_variable(
    name='layer3_biases',
    shape=[LAYER3_NODES],
    initializer=tf.constant_initializer(0.1),
    trainable=True)
y = tf.nn.relu_layer(y_drop, layer3_weights, layer3_biases, name='layer3')
y_drop = tf.nn.dropout(y, keep_prob, name="layer3_drop")
output_weights = tf.get_variable(
    name='output_weights',
    shape=[LAYER2_NODES, 10],
    initializer=default_initializer(),
    trainable=True)
output_biases = tf.get_variable(
    name='output_biases',
    shape=[10],
    initializer=tf.constant_initializer(0.1),
    trainable=True)
y = tf.add(tf.matmul(y_drop, output_weights), output_biases, 'output')

tf.summary.histogram('layer1_weights', layer1_weights)
tf.summary.histogram('layer2_weights', layer2_weights)
tf.summary.histogram('layer3_weights', layer3_weights)
tf.summary.histogram('output_weights', output_weights)

# 定义loss
regularizer = tf.contrib.layers.l2_regularizer(LAMBDA)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)) +\
    regularizer(layer1_weights) +\
    regularizer(layer2_weights) + regularizer(layer3_weights) + \
    regularizer(output_weights)
tf.summary.scalar('loss', loss)
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# 定义准确率
accuracy = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(y, 1, output_type=tf.int32), y_), tf.float32))
tf.summary.scalar('accuracy', accuracy)

# 读取cifar10数据
dict_data, dict_label = cifar10_reader.read('cifar-10-batches-py')
test_data, test_label = cifar10_reader.read('cifar-10-batches-py', False)
test_data = test_data[0:500]
test_label = test_label[0:500]
totallist = list(range(50000))

# 开始会话
sess = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./train', sess.graph, flush_secs=5)
test_writer = tf.summary.FileWriter('./test', flush_secs=5)
sess.run(tf.global_variables_initializer())

# 加载预训练的网络参数
weights = np.load('vgg16_weights.npz')
keys = sorted(weights.keys())
for i, k in enumerate(keys):
    # sess.run(parameters_vgg16[i].assign(weights[k]))
    if i < 26:
        parameters_vgg16[i].load(weights[k], sess)

# 开始训练
for i in range(EPOCH):
    randomlist = random.sample(totallist, BATCH_SIZE)
    x_train = dict_data[randomlist]
    y_train = dict_label[randomlist]
    sess.run(train_step, {x: x_train, y_: y_train, keep_prob: DROP_OUT_RATE})
    train_summary, train_accuracy = sess.run([merged, accuracy], {
        x: x_train,
        y_: y_train,
        keep_prob: 1
    })
    train_writer.add_summary(train_summary, i)
    test_summary, test_accuracy = sess.run([merged, accuracy], {
        x: test_data,
        y_: test_label,
        keep_prob: 1
    })
    test_writer.add_summary(test_summary, i)
    print('step %d,train accuracy is %f, test accuracy is %f' %
          (i, train_accuracy, test_accuracy))

predict_y = sess.run(y, feed_dict={x: [test_data[0]], y: np.zeros((1, 10))})
print(predict_y)
