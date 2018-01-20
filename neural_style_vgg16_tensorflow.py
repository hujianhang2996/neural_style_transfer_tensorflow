import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.preprocessing.image import load_img
import skimage.transform as skimgtr
import imageio
import vgg16_inference as NSVI

default_initializer = tf.contrib.layers.xavier_initializer
weight_file = 'vgg16_weights.npz'
content_img_file = 'city.jpg'
style_img_file = 'vago.jpg'
content_weight = 0.025
style_weight = 1
total_variation_weight = 1
learningRate = 0.01
mean = np.array([[[[123.68000031, 116.77899933, 103.93900299]]]])

width, height = load_img(content_img_file).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)


def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def content_loss(base, combination):
    return K.sum(K.square(combination - base))


def total_variation_loss(x):
    assert K.ndim(x) == 4
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] -
                 x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] -
                 x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


if __name__ == '__main__':
    parameters_vgg16 = []
    layerName_list_vgg16 = iter(['block1_conv1',
                                 'block1_conv2',
                                 'block1_pool',
                                 'block2_conv1',
                                 'block2_conv2',
                                 'block2_pool',
                                 'block3_conv1',
                                 'block3_conv2',
                                 'block3_conv3',
                                 'block3_pool',
                                 'block4_conv1',
                                 'block4_conv2',
                                 'block4_conv3',
                                 'block4_pool',
                                 'block5_conv1',
                                 'block5_conv2',
                                 'block5_conv3',
                                 'block5_pool'])

    x_content = tf.placeholder(
        tf.float32, [1, img_nrows, img_ncols, 3], 'Content_Input')
    x_style = tf.placeholder(
        tf.float32, [1, img_nrows, img_ncols, 3], 'Style_Input')
    x_generate = tf.Variable(tf.truncated_normal(
        [1, img_nrows, img_ncols, 3]), name='Generated_image')
    x = K.concatenate([x_content, x_style, x_generate], axis=0)

    _, y_block_conv1, y_block5_conv2 = NSVI.inference(
        x, layerName_list_vgg16, parameters_vgg16)

    loss = 0
    layer_features = y_block5_conv2
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(base_image_features,
                                          combination_features)

    for layer in y_block_conv1:
        style_reference_features = layer[1, :, :, :]
        combination_features = layer[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(y_block_conv1)) * sl
    loss += total_variation_weight * total_variation_loss(x_generate)
    train_step = tf.train.AdamOptimizer(learningRate).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    weights = np.load(weight_file)
    keys = sorted(weights.keys())
    for i, k in enumerate(keys):
        if i < 26:
            # sess.run(parameters_vgg16[i].assign(weights[k]))
            parameters_vgg16[i].load(weights[k], sess)

    content_img = imageio.imread(content_img_file)
    content_img = np.array(
        [skimgtr.resize(content_img, (img_nrows, img_ncols))]) - mean
    # imageio.imwrite('content_resize.jpg',content_img[0])

    style_img = imageio.imread(style_img_file)
    style_img = np.array(
        [skimgtr.resize(style_img, (img_nrows, img_ncols))]) - mean
    #imageio.imwrite('style_resize.jpg', style_img[0])

    # sess.run(x_generate.initializer)
    # sess.run(x_generate.assign(content_img))
    x_generate.load(content_img, sess)
    for i in range(100):
        loss_trained, _ = sess.run(
            [loss, train_step], {x_content: content_img, x_style: style_img})
        print('step %d, loss is %f' % (i, loss_trained))
        if (i + 1) % 10 == 0:
            generated = sess.run(x_generate)
            generated += mean
            imageio.imwrite('generated_%d.jpg' % (i), generated[0])
