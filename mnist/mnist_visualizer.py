# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def main():
    mnist = read_data_sets('/Users/priyanshusingh/Documents/tfenv', one_hot=True)
    train_images = mnist.train.images
    print(train_images.shape)
    x_image = train_images[0].reshape([28,28])
    # print(x_image.shape)
    plt.gray()
    plt.imshow(x_image)
    plt.show()
    sess = tf.InteractiveSession()
    x_image_rotated = tf.contrib.image.rotate(x_image, 6.14)
    # x_image_rotated = x_image_rotated.reshape([28,28])
    print(type(x_image_rotated.eval()))
    # x_image_rotated = tf.reshape(x_image, [28,28])
    # print(x_image_rotated.shape)
    plt.gray()
    plt.imshow(x_image_rotated.eval())
    plt.show()

if __name__ == "__main__":
    main()
