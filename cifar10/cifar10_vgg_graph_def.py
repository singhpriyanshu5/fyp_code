import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import collections
import time
import os

import scipy

import cifar10_input
from keras import backend as K

initial_learning_rate = 0.1
BATCH_SIZE = 100

def vgg(inputs, is_train_phase, drop_prob=1.0):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):

      net = slim.conv2d(inputs, 64, [3,3], stride=[1,1], padding="SAME")
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase)
      net = slim.dropout(net, 0.3)

      net = slim.conv2d(inputs, 64, [3,3], stride=[1,1], padding="SAME")
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase)

      net = slim.max_pool2d(net, [2, 2], stride=[2,2], padding="SAME")

      net = slim.conv2d(inputs, 128, [3,3], stride=[1,1], padding="SAME")
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase)
      net = slim.dropout(net, 0.4)

      net = slim.conv2d(net, 128, [3,3], stride=[1,1], padding="SAME")
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase)

      net = slim.max_pool2d(net, [2, 2], stride=[2,2], padding="SAME")

      net = slim.conv2d(inputs, 256, [3,3], stride=[1,1], padding="SAME")
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase)
      net = slim.dropout(net, 0.4)


      net = slim.conv2d(net, 256, [3,3], stride=[1,1], padding="SAME")
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase)
      net = slim.dropout(net, 0.4)

      net = slim.conv2d(net, 256, [3,3], stride=[1,1], padding="SAME")
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase)
      net = slim.max_pool2d(net, [2, 2], stride=[2,2], padding="SAME")

      net = slim.conv2d(inputs, 512, [3,3], stride=[1,1], padding="SAME")
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase)
      net = slim.dropout(net, 0.4)

      net = slim.conv2d(net, 512, [3,3], stride=[1,1], padding="SAME")
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase)
      net = slim.dropout(net, 0.4)

      net = slim.conv2d(net, 512, [3,3], stride=[1,1], padding="SAME")
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase)
      net = slim.max_pool2d(net, [2, 2], stride=[2,2], padding="SAME")

      net = slim.conv2d(inputs, 512, [3,3], stride=[1,1], padding="SAME")
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase)
      net = slim.dropout(net, 0.4)

      net = slim.conv2d(net, 512, [3,3], stride=[1,1], padding="SAME")
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase)
      net = slim.dropout(net, 0.4)

      net = slim.conv2d(net, 512, [3,3], stride=[1,1], padding="SAME")
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase)
      net = slim.max_pool2d(net, [2, 2], stride=[2,2], padding="SAME")
      net = slim.dropout(net, 0.5)

      net = slim.fully_connected(slim.flatten(net), 512)
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase)
      net = slim.dropout(net, 0.5)
      net = slim.fully_connected(net, 10, activation_fn=tf.nn.softmax)
    return net

# Input layer
x  = tf.placeholder(tf.float32, [None,32,32,3], name='x')
y_ = tf.placeholder(tf.int32, [None],  name='y_')

# Dropout
drop_prob  = tf.placeholder(tf.float32)
is_train_phase = tf.placeholder(tf.bool)

# Network output
y = vgg(x, is_train_phase, drop_prob)

#get the global step
global_step = tf.train.get_or_create_global_step()
lr = tf.train.exponential_decay(initial_learning_rate, global_step, 25*int(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/BATCH_SIZE), 1e-6, staircase=True)

# Loss/Eval functions
loss = tf.losses.sparse_softmax_cross_entropy(y_, y)
correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1),dtype=tf.int32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Optimizer
#train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
  train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)


output_fld = './media/'

sess = K.get_session()
tf.train.write_graph(sess.graph.as_graph_def(),output_fld, 'model_vgg.pbtxt', as_text=True) 
