import tensorflow as tf
import numpy as np
import time
import collections

from tensorflow.examples.tutorials.mnist import input_data

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from gcnn_lib.grid_graph import grid_graph
from gcnn_lib.coarsening import coarsen
from gcnn_lib.coarsening import lmaxX
from gcnn_lib.coarsening import perm_data
from gcnn_lib.coarsening import lmaxX
from gcnn_lib.coarsening import rescale_L

import cifar10_input

DATA_DIR = "./data"
# mnist = input_data.read_data_sets("data/", one_hot=False)

cifar10_input.maybe_download_and_extract(DATA_DIR)

# train_data = mnist.train.images.astype(np.float32)
train_data, train_labels = cifar10_input.inputs(False, os.path.join(DATA_DIR, 'cifar-10-batches-bin'), cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

test_data, test_labels = cifar10_input.inputs(True, os.path.join(DATA_DIR, 'cifar-10-batches-bin'), cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)
####### test_data = tf.contrib.layers.flatten(test_data)


#test_data = tf.Session().run(test_data)
#rotate each image by a random angle

radians_vector = 6.18*(np.random.random_sample(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,)) + 0.1
train_data = tf.reshape(train_data, [-1,24,24,3])
train_data = tf.contrib.image.rotate(train_data, radians_vector)

#Do we need the reshape back to 784 in this case since we are converting to 28,28
#before passing to lenet

######## train_data = tf.contrib.layers.flatten(train_data)

# Construct graph
grid_side = 24
# num_channels = 3
number_edges = 8
metric = 'euclidean'
A = grid_graph(grid_side,number_edges,metric) # create graph of Euclidean grid

# Compute coarsened graphs
coarsening_levels = 4
num_vertices, L, perm = coarsen(A, coarsening_levels)
# Compute max eigenvalue of graph Laplacians
lmax = []
for i in range(coarsening_levels):
    lmax.append(lmaxX(L[i]))
print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))

class Graph_ConvNet_LeNet5(object):

    # Constructor
    def __init__(self, net_parameters):

        print('Graph ConvNet: LeNet5')

        # parameters
        D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F, NUM_CHANNELS = net_parameters

        # init
        self.WCL1 = self.init_weights([CL1_K*NUM_CHANNELS, CL1_F],CL1_K*NUM_CHANNELS,CL1_F)
        self.bCL1 = tf.zeros([CL1_F],tf.float32)
        self.CL1_F = CL1_F; self.CL1_K = CL1_K
        self.WCL2 = self.init_weights([CL2_K*CL1_F,CL2_F],CL2_K*CL1_F,CL2_F)
        self.bCL2 = tf.zeros([CL2_F],tf.float32)
        self.CL2_F = CL2_F; self.CL2_K = CL2_K
        self.WFC1 = self.init_weights([CL2_F*(D//16),FC1_F],CL2_F*(D//16),FC1_F)
        self.bFC1 = tf.zeros([FC1_F],tf.float32)
        self.WFC2 = self.init_weights([FC1_F,FC2_F],FC1_F,FC2_F)
        self.bFC2 = tf.zeros([FC2_F],tf.float32)

        # Variables for the computational graph
        self.WCL1 = tf.Variable(self.WCL1)
        self.bCL1 = tf.Variable(self.bCL1)
        self.WCL2 = tf.Variable(self.WCL2)
        self.bCL2 = tf.Variable(self.bCL2)
        self.WFC1 = tf.Variable(self.WFC1)
        self.bFC1 = tf.Variable(self.bFC1)
        self.WFC2 = tf.Variable(self.WFC2)
        self.bFC2 = tf.Variable(self.bFC2)


    def init_weights(self, shape, Fin, Fout):

        scale = tf.sqrt( 2.0/ (Fin+Fout) )
        W = tf.random_uniform( shape, minval=-scale, maxval=scale )
        return W


    def graph_conv_cheby(self, x, W, L, lmax, Fout, K):

        # parameters
        # B = batch size
        # V = nb vertices
        # Fin = nb input features
        # Fout = nb output features
        # K = Chebyshev order & support size
        B, V, Fin = x.get_shape(); B, V, Fin = int(B), int(V), int(Fin)
        print("fin", Fin)

        # rescale Laplacian
        lmax = lmaxX(L)
        L = rescale_L(L, lmax)

        # scipy sparse matric of L
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)

        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # V x Fin x B
        x0 = tf.reshape(x0, [V, Fin*B])       # V x Fin*B
        x = tf.expand_dims(x0, 0)             # 1 x V x Fin*B
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)        # 1 x V x Fin*B
            return tf.concat([x, x_], 0)      # K x V x Fin*B
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0
            x = concat(x, x2)                 # M x Fin*B
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, V, Fin, B])     # K x V x Fin x B
        x = tf.transpose(x, perm=[3,1,2,0])   # B x V x Fin x K
        print("xxxx",x)
        x = tf.reshape(x, [B*V, Fin*K])       # B*V x Fin*K

        # Compose linearly Fin features to get Fout features
        x = tf.matmul(x, W)                   # B*V x Fout
        x = tf.reshape(x, [B, V, Fout])       # B x V x Fout

        return x


    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool_(self, x, p):
        if p > 1:
            x = tf.expand_dims(x, 3)  # B x V x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # B x V/p x F
        else:
            return x


    def forward(self, x, d, L, lmax):

        # Graph CL1
        # x = tf.expand_dims(x, 2)  # B x V x Fin=1
        print("forward x")
        print(x)
        x = self.graph_conv_cheby(x, self.WCL1, L[0], lmax[0], self.CL1_F, self.CL1_K) + self.bCL1
        x = tf.nn.relu(x)
        x = self.graph_max_pool_(x, 4)

        # Graph CL2
        x = self.graph_conv_cheby(x, self.WCL2, L[2], lmax[2], self.CL2_F, self.CL2_K) + self.bCL2
        x = tf.nn.relu(x)
        x = self.graph_max_pool_(x, 4)

        # FC1
        B, V, F = x.get_shape(); B, V, F = int(B), int(V), int(F)
        x = tf.reshape(x, [B, V*F])
        x = tf.matmul(x, self.WFC1) + self.bFC1
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, d)

        # FC2
        x = tf.matmul(x, self.WFC2) + self.bFC2

        return x


    def loss(self, x, x_target, l2_regularization):

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x_target, logits=x)
        loss = tf.reduce_mean(cross_entropy)

        l2_loss = 0.0
        tvars = tf.trainable_variables()
        for var in tvars:
            l2_loss += tf.nn.l2_loss(var)

        loss += l2_regularization* l2_loss

        return loss


    def backward(self, loss, learning_rate, train_size, batch_size):

        batch = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, batch * batch_size, train_size, 0.95,
                staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        backward = optimizer.minimize(loss, global_step=batch)

        return backward


    def evaluation(self, x, x_target):

        predicted_classes = tf.cast( tf.argmax( tf.nn.softmax(x), 1 ), tf.int32 )
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_classes,x_target), tf.float32))

        return 100.0* accuracy

# Delete existing network if exists
try:
    del net
    print('Delete existing network\n')
except NameError:
    print('No existing network to delete\n')

def reshape_and_perm(x):
    x_reshaped = np.reshape(x, [x.shape[0], -1, 3])
    x_reshaped = np.reshape(x, [x.shape[0], -1, 3])
    channel1 = x_reshaped[:,:,0]
    channel1 = x_reshaped[:,:,0]
    channel1 = np.squeeze(channel1)
    print("channel1 shape: {}".format(channel1.shape))
    channel1_perm = perm_data(channel1, perm)

    channel2 = x_reshaped[:,:,1]
    channel2 = x_reshaped[:,:,1]
    channel2 = np.squeeze(channel2)
    print("channel2 shape: {}".format(channel2.shape))
    channel2_perm = perm_data(channel2, perm)

    channel3 = x_reshaped[:,:,1]
    channel3 = x_reshaped[:,:,1]
    channel3 = np.squeeze(channel3)
    print("channel3 shape: {}".format(channel3.shape))
    channel3_perm = perm_data(channel3, perm)

    stacked_data = np.dstack((channel1_perm, channel2_perm, channel3_perm))
    print("stacked data shape: {}".format(stacked_data.shape))

    # stacked_data = np.reshape(stacked_data, [stacked_data.shape[0], -1])

    return stacked_data

# network parameters
D = num_vertices[0]

CL1_F = 32
CL1_K = 25
CL2_F = 64
CL2_K = 25
FC1_F = 512
FC2_F = 10
NUM_CHANNELS = 3 #Number of channels in each input image (RGB in this case)
net_parameters = [D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F, NUM_CHANNELS]

# instantiate the object net of the class
net = Graph_ConvNet_LeNet5(net_parameters)

print("net parameters D: {}".format(D))

# learning parameters
learning_rate = 0.05
dropout_value = 0.2
l2_regularization = 5e-4
batch_size = 100
num_epochs = 250
train_size = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
num_channels = 3
nb_iter = int(num_epochs * train_size) // batch_size
print('num_epochs=',num_epochs,', train_size=',train_size,', nb_iter=',nb_iter)


# computational graph
x = tf.placeholder(tf.float32, (batch_size, D, num_channels))
x_target = tf.placeholder(tf.int32, (batch_size))
d = tf.placeholder(tf.float32)

x_score = net.forward(x,d,L,lmax)
loss = net.loss(x_score,x_target,l2_regularization)
backward = net.backward(loss,learning_rate,train_size,batch_size)
evaluation = net.evaluation(x_score,x_target)


# train
init = tf.global_variables_initializer()

tr_cnn = open("logs/cifar10_gcnn.txt",'w')
# loop over epochs
with tf.Session() as sess:
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tf.train.start_queue_runners(sess=sess)

    train_d,train_l,test_d,test_l = sess.run([train_data,train_labels,test_data,test_labels])
    train_d,train_l,test_d,test_l = sess.run([train_data,train_labels,test_data,test_labels])

    train_d = reshape_and_perm(train_d)
    test_d = reshape_and_perm(test_d)

    # train_d = np.reshape(stacked_data, [stacked_data.shape[0], -1])



    # train_d = perm_data(train_d, perm)
    # test_d = perm_data(test_d, perm)

    print("after permutating the data")
    print(train_d.shape)
    print(test_d.shape)

    del perm
    indices = collections.deque()
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        # reshuffle
        indices.extend(np.random.permutation(train_d.shape[0])) # rand permutation

        # reset time
        t_start = time.time()

        # extract batches
        running_loss = 0.0
        running_accuray = 0
        running_total = 0
        while len(indices) >= batch_size:

        # extract batches
            batch_idx = [indices.popleft() for i in range(batch_size)]
            batch_x, batch_y = train_d[batch_idx,:,:], train_l[batch_idx]
            batch_y = np.array(batch_y,np.int32)
            if type(batch_x) is not np.ndarray:
                batch_x = batch_x.toarray()  # convert to full matrices if sparse

            # run computational graph
            _,acc_train,loss_train = sess.run([backward,evaluation,loss], feed_dict={x: batch_x, x_target: batch_y, d: dropout_value})

            # loss, accuracy
            running_loss += loss_train
            running_accuray += acc_train
            running_total += 1

            # print
            if not running_total%100: # print every x mini-batches
                print('epoch= %d, i= %4d, loss(batch)= %.3f, accuray(batch)= %.2f' % (epoch+1, running_total, loss_train, acc_train))

        t_stop = time.time() - t_start
        log_text = "epoch= {}, loss(train)= {:.3f}, accuracy(train)= {:.3f}, time= {:.3f}".format(epoch+1, running_loss/running_total, running_accuray/running_total, t_stop)
        print(log_text)
        tr_cnn.write(log_text)
        tr_cnn.flush()


        # Test set
        running_accuray_test = 0
        running_total_test = 0
        indices_test = collections.deque()
        indices_test.extend(range(test_d.shape[0]))
        t_start_test = time.time()
        while len(indices_test) >= batch_size:
            batch_idx_test = [indices_test.popleft() for i in range(batch_size)]
            batch_x_test, batch_y_test = test_d[batch_idx_test,:,:], test_l[batch_idx_test]
            batch_y_test = np.array(batch_y_test,np.int32)
            if type(batch_x_test) is not np.ndarray:
                batch_x_test = batch_x_test.toarray()  # convert to full matrices if sparse
            acc_test = sess.run(evaluation, feed_dict={x: batch_x_test, x_target: batch_y_test, d: 1.0})
            running_accuray_test += acc_test
            running_total_test += 1
        t_stop_test = time.time() - t_start_test
        log_text = "Accuracy(test) = {:.3f}, time= {:.3f}".format(running_accuray_test / running_total_test, t_stop_test)
        print(log_text)
        tr_cnn.write(log_text)
        tr_cnn.flush()
    tr_cnn.close()
    coord.request_stop()
    coord.join(threads)
