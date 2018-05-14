import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt
#from utils import Plot
from network import *
np.random.seed(7)

## Training parameters

batch_size = 128
iterations = int(55000/batch_size)
val_iterations = int(5000/batch_size)
epochs=25

# ### Getting MNIST data

mnist = input_data.read_data_sets("data/", one_hot=False)
print(mnist.train.images.shape)
print(mnist.validation.images.shape)
print(mnist.test.images.shape)
# ### Setting up the computation graph

tf.reset_default_graph()

#This initializaer is used to initialize all the weights of the network.
initializer = tf.truncated_normal_initializer(stddev=0.02)

#These three placeholders are used for input into the cnn,  generator and discriminator, respectively.
x_in = tf.placeholder(shape=[None, 28, 28, 1],dtype=tf.float32, name='c_in') #Real images
y_ = tf.placeholder(shape=[None],dtype=tf.int32, name='c_gt') #Ground truth for cnn and discriminator

Cx = cnn(x_in, initializer, batch_size) #Pass image through cnn and get fc layer and output

#These functions together define the optimization objective of the CNN+GAN.
c_loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Cx, labels=y_)) # for pretraining

tvars = tf.trainable_variables()

#The below code is responsible for applying gradient descent to update the GAN.
trainerC = tf.train.AdamOptimizer(learning_rate=1e-4)

c_grads1 = trainerC.compute_gradients(c_loss1,tvars) #Only update the weights for the cnn.

correct_prediction = tf.equal(tf.cast(tf.argmax(Cx,1),dtype=tf.int32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

update_C1 = trainerC.apply_gradients(c_grads1)

writer = tf.summary.FileWriter('logs/', graph=tf.get_default_graph())

# ### Training
model_directory='./models'
saver = tf.train.Saver()

max_val_acc = 0
if not os.path.exists(model_directory):
        os.makedirs(model_directory)

with tf.Session(config = tf.ConfigProto(log_device_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
            for i in range(iterations):
                xs,ys = mnist.train.next_batch(batch_size) #Draw a sample batch from MNIST dataset.
                #rotate each image by a random angle
                xs = tf.reshape(xs, [-1,28,28,1])
                #radians_vector = 6.18*(np.random.random_sample(batch_size,)) + 0.1
                #xs = tf.contrib.image.rotate(xs, radians_vector)
                xs = (xs - 0.5) * 2.0 #Transform it to be between -1 and 1
                # xs = np.lib.pad(xs, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1)) #Pad the images so the are 32x32
                sess.run(update_C1,feed_dict={x_in:xs.eval(), y_:ys}) #Update the cnn
            xs = mnist.validation.images
            xs = (np.reshape(xs,[xs.shape[0],28,28,1]) - 0.5) * 2.0 #Transform it to be between -1 and 1
            # xs = np.lib.pad(xs, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1)) #Pad the images so the are 32x32
            ys = mnist.validation.labels
            loss, acc = sess.run([c_loss1,accuracy],feed_dict={x_in:xs,y_:ys})
            print("Epoch: {} Loss: {:.2f} Accuracy: {:.2f}".format(epoch, loss, acc*100))
            if acc > max_val_acc:
                max_val_acc = acc
                saver.save(sess, model_directory+'/CNN.ckpt')

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #Reload the model.
    print("Loading Model...")
    ckpt = tf.train.get_checkpoint_state(model_directory)
    saver.restore(sess,ckpt.model_checkpoint_path)
    xs = mnist.test.images
    xs = (np.reshape(xs,[xs.shape[0],28,28,1]) - 0.5) * 2.0 #Transform it to be between -1 and 1
    # xs = np.lib.pad(xs, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1)) #Pad the images so the are 32x32
    ys = mnist.test.labels
    loss, acc = sess.run([c_loss1,accuracy],feed_dict={x_in:xs,y_:ys})
    print("Test Loss: {:.2f} \t Test Acc: {:.2f}".format(loss,acc*100))
