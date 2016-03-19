################################################################################
#Michael Guerzhoy, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
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


# parameters
img_dim = 64
input_dim = 32 ** 2
output_dim = 6

# weights and biases
weights = {
    'conv1': tf.Variable(tf.random_normal([11, 11, 3, 96])),
    'conv2': tf.Variable(tf.random_normal([5, 5, 48, 256])),
    'conv3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'conv4': tf.Variable(tf.random_normal([3, 3, 192, 384])),
    'conv5': tf.Variable(tf.random_normal([3, 3, 192, 256])),
    'fc6': tf.Variable(tf.random_normal([256, 1024])),
    'fc7': tf.Variable(tf.random_normal([1024, 1024])),
    'fc8': tf.Variable(tf.random_normal([1024, output_dim]))
}

biases = {
    'conv1': tf.Variable(tf.random_normal([96])),
    'conv2': tf.Variable(tf.random_normal([256])),
    'conv3': tf.Variable(tf.random_normal([384])),
    'conv4': tf.Variable(tf.random_normal([384])),
    'conv5': tf.Variable(tf.random_normal([256])),
    'fc6': tf.Variable(tf.random_normal([1024])),
    'fc7': tf.Variable(tf.random_normal([1024])),
    'fc8': tf.Variable(tf.random_normal([output_dim]))
}


#input, used a placeholder
network_input = tf.placeholder(tf.float32, [1, 64 ** 2 * 3])
x = tf.reshape(network_input, [1, 64, 64, 3])

def zipdict(a, b):
    out = {}
    for key in a.keys():
        out[key] = (a[key], b[key])
    return out

net_data = zipdict(weights, biases)


# convolution 
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):

    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())




#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = net_data["conv1"][0]
conv1b = net_data["conv1"][1]
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = net_data["conv2"][0]
conv2b = net_data["conv2"][1]
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = net_data["conv3"][0]
conv3b = net_data["conv3"][1]
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = net_data["conv4"][0]
conv4b = net_data["conv4"][1]
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = net_data["conv5"][0]
conv5b = net_data["conv5"][1]
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = net_data["fc6"][0]
fc6b = net_data["fc6"][1]
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = net_data["fc7"][0]
fc7b = net_data["fc7"][1]
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
fc8W = net_data["fc8"][0]
fc8b = net_data["fc8"][1]
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)






#prob
#softmax(name='prob'))
prediction = tf.nn.softmax(fc8)
argmax_prediction = tf.argmax(tf.nn.softmax(fc8), 1)

# declare the cost function (negative log likelihood), training step
network_expected = tf.placeholder(tf.float32, [None, output_dim])
NLL = -tf.reduce_sum(network_expected*tf.log(prediction))
train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(NLL)

# score the output vs expected output
correct_prediction = tf.equal(tf.argmax(network_expected,1), tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize the session

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
sess.run(tf.assert_variables_initialized())

