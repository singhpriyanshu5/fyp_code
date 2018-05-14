import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import collections
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=False)

train_data = mnist.train.images.astype(np.float32)
#rotate each image by a random angle
radians_vector = 6.18*(np.random.random_sample(train_data.shape[0],)) + 0.1
images = tf.reshape(train_data, [-1,28,28,1])
images_rotated = tf.contrib.image.rotate(images, radians_vector)
#Do we need the reshape back to 784 in this case since we are converting to 28,28
#before passing to lenet
images_rotated = tf.reshape(images_rotated, [-1, train_data.shape[1]])

train_data = images_rotated

#uncomment the following line to use the rotated test set
test_data = mnist.test.images.astype(np.float32)
radians_vector = 6.18*(np.random.random_sample(test_data.shape[0],)) + 0.1
images_test = tf.reshape(test_data, [-1,28,28,1])
images_rotated_test = tf.contrib.image.rotate(images_test, radians_vector)
test_data = tf.contrib.layers.flatten(images_rotated_test)


val_data = mnist.validation.images.astype(np.float32)
#comment the following line out to use the rotated test set
#test_data = mnist.test.images.astype(np.float32)
train_labels = mnist.train.labels
val_labels = mnist.validation.labels
test_labels = mnist.test.labels

train_data = tf.Session().run(train_data)
#uncomment the following line to use the rotated test set
test_data = tf.Session().run(test_data)

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
x  = tf.placeholder(tf.float32, [None,784], name='x')
y_ = tf.placeholder(tf.int32, [None],  name='y_')
x_image = tf.reshape(x, [-1,28,28,1])

# Dropout
drop_prob  = tf.placeholder(tf.float32)

# Network output
y = lenet(x_image, drop_prob)

# Loss/Eval functions
loss = tf.losses.sparse_softmax_cross_entropy(y_, y)
correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1),dtype=tf.int32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Optimizer
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

# Training/Testing
BATCH_SIZE = 100
TRAIN_ITERATIONS = int(55000/BATCH_SIZE)
TEST_ITERATIONS = int(10000/BATCH_SIZE)
TRAIN_SIZE = train_data.shape[0]
EPOCHS = 30
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  indices = collections.deque()
  tr_cnn = open("logs/cnn_logs.txt",'w')
  for epoch in range(EPOCHS):
    indices.extend(np.random.permutation(TRAIN_SIZE))
    t_start = time.time()
    step_loss, step_acc = 0.0, 0.0
    running_total = 0

    #print("initial index length = {}".format(len(indices))

    while(len(indices) >= BATCH_SIZE):
        batch_idx = [indices.popleft() for i in range(BATCH_SIZE)]
        batch_x, batch_y = train_data[batch_idx,:], train_labels[batch_idx]
        batch_y = np.array(batch_y,np.int32)
        if type(batch_x) is not np.ndarray:
            batch_x = batch_x.toarray()  # convert to full matrices if sparse
        _, l, a = sess.run([train_step, loss, accuracy], feed_dict={x:batch_x, y_: batch_y,
                                                                    drop_prob: 0.5})
        step_loss += l
        step_acc += a
        running_total += 1
        if running_total%100 == 0: # print every x mini-batches
            print('index length= {} epoch= {}, i= {}, loss(batch)= {:.3f}, accuray(batch)= {:.2f}'.format(len(indices), epoch+1, running_total, l, a))
    t_stop = time.time() - t_start
    log_text = "Training Epoch: ({}/{})\tLoss: {:.2f}\tAccuracy: {:.2f}\t Time: {:.2f}".format(epoch+1, EPOCHS,
                                                                   step_loss/running_total,
                                                                   step_acc/running_total,
                                                                   t_stop)
    print(log_text)
    tr_cnn.write(log_text)
    tr_cnn.flush()

    running_total_loss = 0
    running_accuray_test = 0
    running_total_test = 0
    indices_test = collections.deque()
    indices_test.extend(range(test_data.shape[0]))
    t_start_test = time.time()
    while len(indices_test) >= BATCH_SIZE:
        batch_idx_test = [indices_test.popleft() for i in range(BATCH_SIZE)]
        batch_x_test, batch_y_test = test_data[batch_idx_test,:], test_labels[batch_idx_test]
        batch_y_test = np.array(batch_y_test,np.int32)
        if type(batch_x_test) is not np.ndarray:
            batch_x_test = batch_x_test.toarray()  # convert to full matrices if sparse
        l, a = sess.run([loss, accuracy], feed_dict={x: batch_x_test, y_: batch_y_test, drop_prob: 1.0})
        running_total_loss += l
        running_accuray_test += a
        running_total_test += 1
    t_stop_test = time.time() - t_start_test
    log_text = "Testing result=> Loss: {:.2f}\tAccuracy: {:.3f}, time= {:.3f}".format(running_total_loss/running_total_test, running_accuray_test/running_total_test, t_stop_test)
    print(log_text)
    tr_cnn.write(log_text)
    tr_cnn.flush()

  tr_cnn.close()
