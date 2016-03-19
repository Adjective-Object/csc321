from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib

from numpy import random
import tensorflow as tf

import os



img_size = 1024
nhid = 300
num_classes = 6


inputs = tf.placeholder(tf.float32, [None, img_size])

# linear layer 0, input (32x32 -> num hidden)
W0 = tf.Variable(tf.random_normal([img_size, nhid], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

# linear layer 2 (num hidden -> num classes)
W1 = tf.Variable(tf.random_normal([nhid, num_classes], stddev=0.01))
b1 = tf.Variable(tf.random_normal([num_classes], stddev=0.01))

# connecting the layers to their activation functions
layer1 = tf.nn.tanh(tf.matmul(inputs, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1

# softmaxing the output layer & comparing it with the expected values
prediction = tf.nn.softmax(layer2)
expectation = tf.placeholder(tf.float32, [None, num_classes])

NLL = -tf.reduce_sum(expectation*tf.log(prediction))

# add l2 regularization to get nice weights or something
reg = (tf.nn.l2_loss(W0) + tf.nn.l2_loss(b0) +
       tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1))
NLL += 5e-4 * reg

train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(expectation,1), tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

