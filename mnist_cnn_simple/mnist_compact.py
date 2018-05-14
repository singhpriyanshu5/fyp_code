import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=False)

def lenet(inputs, drop_prob=1.0):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
      net = slim.conv2d(inputs, 32, [5,5], stride=[1,1], padding="SAME", scope='conv1')
      net = slim.max_pool2d(net, [2, 2], stride=[2,2], padding="SAME", scope='pool1')
      net = slim.conv2d(net, 64, [5,5], stride=[1,1], padding="SAME", scope='conv2')
      net = slim.max_pool2d(net, [2, 2], stride=[2,2], padding="SAME", scope='pool2')
      net = slim.fully_connected(slim.flatten(net), 1024, scope='fc1')
      net = slim.dropout(net, drop_prob, scope='dropout1')
      net = slim.fully_connected(net, 10, activation_fn=None, scope='fc2')
    return net

# Input layer
x  = tf.placeholder(tf.float32, [None,28,28,1], name='x')
y_ = tf.placeholder(tf.int32, [None],  name='y_')
#x_image = tf.reshape(x, [-1,28,28,1])
#radians_vector = 6.18*(tf.random_uniform(shape=[x_image.get_shape()[0]], dtype=tf.float32, name='rotation_anlges')) + 0.1
#x_image = tf.contrib.image.rotate(x_image, radians_vector)

# Dropout
drop_prob  = tf.placeholder(tf.float32)

# Network output
y = lenet(x, drop_prob)

# Loss/Eval functions
loss = tf.losses.sparse_softmax_cross_entropy(y_, y)
correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1),dtype=tf.int32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Optimizer
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

# Training/Testing
BATCH_SIZE = 128
TRAIN_ITERATIONS = int(55000/BATCH_SIZE)
TEST_ITERATIONS = int(10000/BATCH_SIZE)
EPOCHS = 25
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  radians_vector =  6.18*(tf.random_uniform(shape=[BATCH_SIZE], dtype=tf.float32, name='rotation_angles')) + 0.1
  for epoch in range(EPOCHS):
    step_loss, step_acc = 0.0, 0.0
    for i in range(TRAIN_ITERATIONS):
      batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
      batch_xs = (batch_xs - 0.5)*2
      x_image = tf.reshape(batch_xs, [BATCH_SIZE,28,28,1])
      x_image = tf.contrib.image.rotate(x_image, radians_vector)
      _, l, a = sess.run([train_step, loss, accuracy], feed_dict={x: x_image.eval(), y_: batch_ys,
                                                                  drop_prob: 0.5})
      step_loss += l
      step_acc += a
    print("Training Epoch: ({}/{})\tLoss: {:.2f}\tAccuracy: {:.2f}".format(epoch, EPOCHS,
                                                                   step_loss/TRAIN_ITERATIONS,
                                                                   (step_acc/TRAIN_ITERATIONS)*100))
  batch_xs, batch_ys = mnist.test.images, mnist.test.labels
  batch_xs = (batch_xs - 0.5)*2
  batch_xs = tf.reshape(batch_xs, [-1,28,28,1])
  l, a = sess.run([loss, accuracy], feed_dict={x: batch_xs, y_: batch_ys, drop_prob: 1.0})
  print("Testing Result: Loss: {:.2f}\tAccuracy: {:.2f}".format(l, a*100))
