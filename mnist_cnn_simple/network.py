import tensorflow as tf
import tensorflow.contrib.slim as slim

def cnn(input_images, weights_initializer, batch_size, keep_prob=1.0):
    biases_initializer = tf.constant_initializer(0.1)
    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable(name='kernel', shape=[5,5,1,32], initializer=weights_initializer, dtype=tf.float32)
        conv = tf.nn.conv2d(input_images, kernel, strides=[1,1,1,1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[32], initializer=biases_initializer, dtype=tf.float32)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable(name='kernel', shape=[5,5,32,64], initializer=weights_initializer, dtype=tf.float32)
        conv = tf.nn.conv2d(pool1, kernel, strides=[1,1,1,1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[64], initializer=biases_initializer, dtype=tf.float32)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')

    with tf.variable_scope('fc1') as scope:
        #reshape = tf.reshape(pool2, [batch_size,-1])
        #dim = reshape.get_shape()[1].value
        flatten = tf.contrib.layers.flatten(pool2)
        dim = flatten.get_shape()[1].value
        weights = tf.get_variable(name='weights', shape=[dim,1024], initializer=weights_initializer, dtype=tf.float32)
        biases = tf.get_variable(name='biases', shape=[1024], initializer=biases_initializer, dtype=tf.float32)
        pre_activation = tf.nn.bias_add(tf.matmul(flatten, weights), biases)
        fc1 = tf.nn.relu(pre_activation, name=scope.name)

    dropout = tf.nn.dropout(fc1, keep_prob, name='dropout')

    with tf.variable_scope('fc2') as scope:
        weights = tf.get_variable(name='weights', shape=[1024,10], initializer=weights_initializer, dtype=tf.float32)
        biases = tf.get_variable(name='biases', shape=[10], initializer=biases_initializer, dtype=tf.float32)
        pre_activation = tf.nn.bias_add(tf.matmul(dropout, weights), biases)
        fc2 = tf.nn.relu(pre_activation, name=scope.name)

    return fc2

# #CNN to perform classification
# def cnn(x, initializer, keep_prob=1.0, reuse=False):
#
#     cnn1 = slim.convolution2d(x,32,[5,5],stride=[1,1],padding="SAME",\
#         biases_initializer=None,activation_fn=tf.nn.relu,\
#         ,scope='c_conv1',weights_initializer=initializer)
#
#     cnn2 = slim.max_pool2d(cnn1,[2,2],stride=[2,2], padding="SAME", scope='c_pool1')
#
#     cnn3 = slim.convolution2d(cnn2,64,[5,5],stride=[1,1],padding="SAME",\
#         biases_initializer=None,activation_fn=tf.nn.relu,\
#         ,scope='c_conv2',weights_initializer=initializer)
#
#     cnn4 = slim.max_pool2d(cnn3,[2,2],stride=[2,2], padding="SAME", scope='c_pool2')
#
#     cnn5 = slim.fully_connected(slim.flatten(cnn4),1024,activation_fn=tf.nn.relu,\
#         ,scope='c_fc1', weights_initializer=initializer)
#
#     cnn6 = slim.dropout(cnn5,keep_prob, scope='c_d1')
#
#     c_out = slim.fully_connected(cnn6,10,activation_fn=None,\
#         ,scope='c_out', weights_initializer=initializer)
#
#     return cnn6, c_out
