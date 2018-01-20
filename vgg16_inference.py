import tensorflow as tf
default_initializer = tf.contrib.layers.xavier_initializer


def conv_layer(input, w_shape, name, inference_reuse):
    with tf.variable_scope(name, reuse=inference_reuse):
        weights = tf.get_variable(
            name='weights', shape=w_shape, initializer=default_initializer(), trainable=False)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)
        biases = tf.get_variable(name='biases', shape=[
                                 w_shape[-1]], initializer=default_initializer(), trainable=False)
        y = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
            input, weights, [1, 1, 1, 1], 'SAME'), biases))

    return y, weights, biases


def pool_layer(input, name, inference_reuse):
    with tf.variable_scope(name, reuse=inference_reuse):
        y = tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    return y


def fc_layer(input, name, n_out, inference_reuse, actfun=tf.nn.relu_layer):
    n_in = input.get_shape()[-1].value
    with tf.variable_scope(name, reuse=inference_reuse):
        weights = tf.get_variable(name='weights', shape=[
                                  n_in, n_out], initializer=default_initializer(), trainable=False)
        biases = tf.get_variable(name='biases', shape=[
                                 n_out], initializer=tf.constant_initializer(0.1), trainable=False)
        y = actfun(input, weights, biases, name=name)
        return y, weights, biases


def inference(input_tensor, layerName_list, parameters, inference_reuse=False):
    # block 1 -- outputs 112x112x64
    y_block1_conv1, weights, biases = conv_layer(
        input_tensor, [3, 3, 3, 64], next(layerName_list), inference_reuse)
    parameters += [weights, biases]
    y, weights, biases = conv_layer(
        y_block1_conv1, [3, 3, 64, 64], next(layerName_list), inference_reuse)
    parameters += [weights, biases]
    y = pool_layer(y, next(layerName_list), inference_reuse)
    # block 2 -- outputs 56x56x128
    y_block2_conv1, weights, biases = conv_layer(
        y, [3, 3, 64, 128], next(layerName_list), inference_reuse)
    parameters += [weights, biases]
    y, weights, biases = conv_layer(
        y_block2_conv1, [3, 3, 128, 128], next(layerName_list), inference_reuse)
    parameters += [weights, biases]
    y = pool_layer(y, next(layerName_list), inference_reuse)
    # block 3 -- outputs 28x28x256
    y_block3_conv1, weights, biases = conv_layer(
        y, [3, 3, 128, 256], next(layerName_list), inference_reuse)
    parameters += [weights, biases]
    y, weights, biases = conv_layer(
        y_block3_conv1, [3, 3, 256, 256], next(layerName_list), inference_reuse)
    parameters += [weights, biases]
    y, weights, biases = conv_layer(
        y, [3, 3, 256, 256], next(layerName_list), inference_reuse)
    parameters += [weights, biases]
    y = pool_layer(y, next(layerName_list), inference_reuse)
    # block 4 -- outputs 14x14x512
    y_block4_conv1, weights, biases = conv_layer(
        y, [3, 3, 256, 512], next(layerName_list), inference_reuse)
    parameters += [weights, biases]
    y, weights, biases = conv_layer(
        y_block4_conv1, [3, 3, 512, 512], next(layerName_list), inference_reuse)
    parameters += [weights, biases]
    y, weights, biases = conv_layer(
        y, [3, 3, 512, 512], next(layerName_list), inference_reuse)
    parameters += [weights, biases]
    y = pool_layer(y, next(layerName_list), inference_reuse)
    # block 5 -- outputs 7x7x512
    y_block5_conv1, weights, biases = conv_layer(
        y, [3, 3, 512, 512], next(layerName_list), inference_reuse)
    parameters += [weights, biases]
    y_block5_conv2, weights, biases = conv_layer(
        y_block5_conv1, [3, 3, 512, 512], next(layerName_list), inference_reuse)
    parameters += [weights, biases]
    y, weights, biases = conv_layer(
        y_block5_conv2, [3, 3, 512, 512], next(layerName_list), inference_reuse)
    parameters += [weights, biases]
    y_block5_pool = pool_layer(y, next(layerName_list), inference_reuse)
    # flatten
    shp = y_block5_pool.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(y_block5_pool, [-1, flattened_shape], name="reshape")
    # fully connected
    fc6, weights, biases = fc_layer(resh1, "fc6", 4096, inference_reuse)
    parameters += [weights, biases]
    fc6_drop = tf.nn.dropout(fc6, 0.5, name="fc6_drop")

    fc7, weights, biases = fc_layer(fc6_drop, "fc7", 4096, inference_reuse)
    parameters += [weights, biases]
    fc7_drop = tf.nn.dropout(fc7, 0.5, name="fc7_drop")

    y, weights, biases = fc_layer(fc7_drop, "fc8", 1000, inference_reuse)
    parameters += [weights, biases]
    return y_block5_pool, [y_block1_conv1, y_block2_conv1, y_block3_conv1, y_block4_conv1, y_block5_conv1], y_block5_conv2
