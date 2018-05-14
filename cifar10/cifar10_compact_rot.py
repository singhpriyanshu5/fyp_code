import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import collections
import time
import os

import scipy

import cifar10_input
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = "./data"
# mnist = input_data.read_data_sets("data/", one_hot=False)

cifar10_input.maybe_download_and_extract(DATA_DIR)

# train_data = mnist.train.images.astype(np.float32)
train_data, train_labels = cifar10_input.inputs(False, os.path.join(DATA_DIR, 'cifar-10-batches-bin'), cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN+cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_VAL)

test_data, test_labels = cifar10_input.inputs(True, os.path.join(DATA_DIR, 'cifar-10-batches-bin'), cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)
#test_data = tf.Session().run(test_data)
#rotate each image by a random angle

val_data = tf.slice(train_data, [cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, 0, 0, 0], [-1, -1, -1, -1])
val_labels = tf.slice(train_labels, [cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN], [-1])


print('train_data', train_data)
print('train_data only train', train_data[:cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,:,:,:])
print('val_data', val_data)

#radians_vector = 6.18*(np.random.random_sample(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,)) + 0.1
#train_data = tf.reshape(train_data, [-1,24,24,3])
#train_data = tf.contrib.image.rotate(train_data[:cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,:,:,:], radians_vector)
#Do we need the reshape back to 784 in this case since we are converting to 28,28
#before passing to lenet

#train_data = tf.contrib.layers.flatten(train_data)

#radians_vector_val = 6.18*(np.random.random_sample(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_VAL,)) + 0.1
#val_data = tf.reshape(val_data, [-1,24,24,3])
#val_data = tf.contrib.image.rotate(val_data, radians_vector_val)

#val_data = tf.contrib.layers.flatten(val_data)

#radians_vector_test = 6.18*(np.random.random_sample(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,)) + 0.1
#test_data = tf.reshape(test_data, [-1,24,24,3])
#test_data = tf.contrib.image.rotate(test_data, radians_vector_test)

#test_data = tf.contrib.layers.flatten(test_data)

#val_data = tf.slice(train_data, [cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, 0], [-1,-1])
#val_labels = tf.slice(train_labels, [cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN], [-1])

#train_data = tf.Session().run(train_data)

# images_rotated = tf.reshape(images_rotated, [-1, train_data.shape[1]])

# train_data = images_rotated
# val_data = mnist.validation.images.astype(np.float32)
# test_data = mnist.test.images.astype(np.float32)
# train_labels = mnist.train.labels
# val_labels = mnist.validation.labels
# test_labels = mnist.test.labels

# train_data = tf.Session().run(train_data)

def lenet(inputs, is_train_phase, drop_prob=1.0):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
      net = slim.conv2d(inputs, 64, [5,5], stride=[1,1], padding="SAME", scope='conv1')
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase, scope='bn1')
      #net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
      net = slim.max_pool2d(net, [3, 3], stride=[2,2], padding="SAME", scope='pool1')
      net = slim.conv2d(net, 64, [5,5], stride=[1,1], padding="SAME", scope='conv2')
      net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train_phase, scope='bn2')
      #net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
      net = slim.max_pool2d(net, [3, 3], stride=[2,2], padding="SAME", scope='pool2')
      net = slim.fully_connected(slim.flatten(net), 384, scope='fc1')
      net = slim.dropout(net, drop_prob, scope='dropout')
      net = slim.fully_connected(net, 192, scope='fc2') 
      net = slim.fully_connected(net, 10, activation_fn=None, scope='fc3')
    return net

# Input layer
x  = tf.placeholder(tf.float32, [None,32,32,3], name='x')
y_ = tf.placeholder(tf.int32, [None],  name='y_')
#x_image = tf.reshape(x, [-1,24,24,3])
#radians_vector = 6.18*(tf.random_uniform(shape=[x_image.get_shape()[0]], dtype=tf.float32, name='rotation_anlges')) + 0.1
#x_image = tf.contrib.image.rotate(x_image, radians_vector)

# Dropout
drop_prob  = tf.placeholder(tf.float32)
is_train_phase = tf.placeholder(tf.bool)

# Network output
y = lenet(x, is_train_phase, drop_prob)

# Loss/Eval functions
loss = tf.losses.sparse_softmax_cross_entropy(y_, y)
correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1),dtype=tf.int32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Optimizer
#train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

model_directory='./models'
saver = tf.train.Saver()

# Training/Testing
BATCH_SIZE = 100
TRAIN_ITERATIONS = int(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/BATCH_SIZE)
TEST_ITERATIONS = int(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL/BATCH_SIZE)
EPOCHS = 100
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  tf.train.start_queue_runners(sess=sess)

  train_d,train_l,test_d,test_l,val_d,val_l = sess.run([train_data,train_labels,test_data,test_labels,val_data,val_labels])

#Rescaling the training data to the range 0 to 1
  train_d_rescaled = []
  for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN):
      train_d_rescaled.append(scipy.misc.bytescale(train_d[i]))
  train_d = np.stack(train_d_rescaled)

#Rescaling the testing data to the range 0 to 1
  test_d_rescaled = []
  for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL):
       test_d_rescaled.append(scipy.misc.bytescale(test_d[i]))
  test_d = np.stack(test_d_rescaled)

#Rescaling the validation data to the range 0 to 1
  val_d_rescaled = []
  for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_VAL):
      val_d_rescaled.append(scipy.misc.bytescale(val_d[i]))
  val_d = np.stack(val_d_rescaled)

  train_d_rotated = []
  degrees_vector = np.random.random_integers(-90, 90, size=(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,))
  for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN):
      x_image = scipy.misc.bytescale(train_d[i])
      x_image_padded = np.lib.pad(x_image, ((7,7),(7,7),(0,0)), 'symmetric')
      x_image_rotated = scipy.misc.imrotate(x_image_padded, degrees_vector[i], 'cubic')
      x_cropped = x_image_rotated[7:39,7:39,:]
      train_d_rotated.append(x_cropped)

  #image_normal = scipy.misc.bytescale(train_d[0])
  #image_rotated = train_d_rotated[0]
  train_d = np.stack(train_d_rotated)
  print("train_d shape after rotating",train_d.shape)

  print("train_data max",np.amax(train_d))
  print("train_data min",np.amin(train_d))
  print("test_data max",np.amax(test_d))
  print("test_data min",np.amin(test_d))
  print("val data max", np.amax(val_d))
  print("val data min", np.amin(val_d))
  #degrees_vector_test = np.random.random_integers(-90, 90, size=(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,))
  #test_d_rotated = []
  #for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL):
  #     x_image = test_d[i]
  #     x_image_padded = np.lib.pad(x_image, ((7,7),(7,7),(0,0)), 'symmetric')
  #     x_image_rotated = scipy.misc.imrotate(x_image_padded, degrees_vector_test[i], 'cubic')
  #     x_cropped = x_image_rotated[7:39,7:39,:]
  #     test_d_rotated.append(x_cropped)
  #test_d = np.stack(test_d_rotated)
  #print("test_d shape after rotating",test_d.shape)

  #degrees_vector_val = np.random.random_integers(-90, 90, size=(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_VAL,))
  #val_d_rotated = []
  #for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_VAL):
  #    x_image = val_d[i]
  #    x_image_padded = np.lib.pad(x_image, ((7,7),(7,7),(0,0)), 'symmetric')
  #    x_image_rotated = scipy.misc.imrotate(x_image_padded, degrees_vector_val[i], 'cubic')
  #    x_cropped = x_image_rotated[7:39,7:39,:]
  #    val_d_rotated.append(x_cropped)
  #val_d = np.stack(val_d_rotated)
  #print("val_d shape after rotating",val_d.shape)

#Rotating the testing and validation data
  #test_d_rotated = []
  #val_d_rotated = []
  #degrees_vector_test = np.random.random_integers(-90, 90, size=(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,))
  #for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL):
  #    x_image = test_d[i]
  #    x_image_padded = np.lib.pad(x_image, ((7,7),(7,7),(0,0)), 'symmetric')
  #    x_image_rotated = scipy.misc.imrotate(x_image_padded, -90, 'cubic')
  #    x_cropped = x_image_rotated[7:39,7:39,:]
  #    test_d_rotated.append(x_cropped)
  #test_d = np.stack(test_d_rotated)
  #print("test_d shape after rotating", test_d.shape)

  #degrees_vector_val = np.random.random_integers(-90, 90, size=(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_VAL,))
  #for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_VAL):
  #    x_image = val_d[i]
  #    x_image_padded = np.lib.pad(x_image, ((7,7),(7,7),(0,0)), 'symmetric')
  #    x_image_rotated = scipy.misc.imrotate(x_image_padded, -90, 'cubic')
  #    x_cropped = x_image_rotated[7:39,7:39,:]
  #    val_d_rotated.append(x_cropped)
  #val_d = np.stack(val_d_rotated)
  #print("val_d shape after rotating", val_d.shape)

  indices = collections.deque()
  TRAIN_SIZE = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  tr_cnn = open("logs/cifar10_logs.txt", 'w')
  max_val_accuracy = 0
  for epoch in range(EPOCHS):
    indices.extend(np.random.permutation(TRAIN_SIZE))
    t_start = time.time()
    step_loss, step_acc = 0.0, 0.0
    running_total = 0

    #print("initial index length = {}".format(len(indices))

    while(len(indices) >= BATCH_SIZE):
        batch_idx = [indices.popleft() for i in range(BATCH_SIZE)]
        batch_x, batch_y = train_d[batch_idx,:,:,:], train_l[batch_idx]
        batch_y = np.array(batch_y,np.int32)
        if type(batch_x) is not np.ndarray:
            batch_x = batch_x.toarray()  # convert to full matrices if sparse
        _, l, a = sess.run([train_step, loss, accuracy], feed_dict={x:batch_x, y_: batch_y, is_train_phase: True, drop_prob: 0.2})
        step_loss += l
        step_acc += a
        running_total += 1
        if running_total%10 == 0: # print every x mini-batches
            print('index length= {} epoch= {}, i= {}, loss(batch)= {:.3f}, accuray(batch)= {:.2f}'.format(len(indices), epoch+1, running_total, l, a))
    t_stop = time.time() - t_start
    log_text = "Training Epoch: ({}/{})\tLoss: {:.2f}\tAccuracy: {:.2f}\t Time: {:.2f}".format(epoch+1, EPOCHS,
                                                                   step_loss/running_total,
                                                                   step_acc/running_total,
                                                                   t_stop)
    print(log_text)
    tr_cnn.write(log_text)
    tr_cnn.flush()

    running_loss_val = 0
    running_accuracy_val = 0
    running_total_val = 0
    indices_val = collections.deque()
    indices_val.extend(range(val_data.shape[0]))
    t_start_val = time.time()
    while len(indices_val) >= BATCH_SIZE:
        batch_idx_val = [indices_val.popleft() for i in range(BATCH_SIZE)]
        batch_x_val, batch_y_val = val_d[batch_idx_val,:], val_l[batch_idx_val]
        batch_y_val = np.array(batch_y_val, np.int32)
        if type(batch_x_val) is not np.ndarray:
            batch_x_val = batch_x_val.toarray()
        l, a = sess.run([loss, accuracy], feed_dict={x: batch_x_val, y_: batch_y_val, is_train_phase: False, drop_prob: 1.0})
        running_loss_val += l
        running_accuracy_val += a
        running_total_val += 1
    t_stop_val = time.time() - t_start_val
    val_acc = running_accuracy_val/running_total_val
    if val_acc > max_val_accuracy:
        max_val_accuracy = val_acc
        saver.save(sess, model_directory+'/CNN.ckpt')
        print("saved model")
    log_text = "Validation result => Loss: {:.2f}\t Accuracy: {:.3f}, time={:.3f}".format(running_loss_val/running_total_val, running_accuracy_val/running_total_val, t_stop_val)
    print(log_text)
    tr_cnn.write(log_text)
    tr_cnn.flush()

  running_loss_test = 0
  running_accuray_test = 0
  running_total_test = 0
  indices_test = collections.deque()
  indices_test.extend(range(test_data.shape[0]))
  t_start_test = time.time()
  ckpt = tf.train.get_checkpoint_state(model_directory)
  saver.restore(sess, ckpt.model_checkpoint_path)
  while len(indices_test) >= BATCH_SIZE:
      batch_idx_test = [indices_test.popleft() for i in range(BATCH_SIZE)]
      batch_x_test, batch_y_test = test_d[batch_idx_test,:], test_l[batch_idx_test]
      batch_y_test = np.array(batch_y_test,np.int32)
      if type(batch_x_test) is not np.ndarray:
          batch_x_test = batch_x_test.toarray()  # convert to full matrices if sparse
      l, a = sess.run([loss, accuracy], feed_dict={x: batch_x_test, y_: batch_y_test, is_train_phase: False, drop_prob: 1.0})
      running_loss_test += l
      running_accuray_test += a
      running_total_test += 1
  t_stop_test = time.time() - t_start_test
  log_text = "Testing result=> Loss: {:.2f}\tAccuracy: {:.3f}, time= {:.3f}".format(running_loss_test/running_total_test, running_accuray_test/running_total_test, t_stop_test)
  print(log_text)

  #loss_normal = sess.run([loss], feed_dict={x: np.expand_dims(image_normal, 0), y_: np.expand_dims(train_l[0], 0), is_train_phase: False, drop_prob: 1.0})
  #loss_rotated = sess.run([loss], feed_dict={x: np.expand_dims(image_rotated, 0), y_: np.expand_dims(train_l[0], 0), is_train_phase: False, drop_prob: 1.0})
  #print("loss_normal: {}, loss_rotated: {}".format(loss_normal, loss_rotated))
  tr_cnn.write(log_text)
  tr_cnn.flush()
  tr_cnn.close()

  coord.request_stop()
  coord.join(threads)
