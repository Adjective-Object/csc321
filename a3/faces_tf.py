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
from scipy.io import loadmat


# helpers for loading face shit
import face_utils

random.seed(555)








lam = 0.00000
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
NLL = -tf.reduce_sum(y_*tf.log(y)+decay_penalty)

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = get_test(M)


def do_train(data, bsize=10):
    for i in range(1000):
        #print i  
        names, training_batch, training_outputs = load_fileset(
            data["training"],
            "training",
            i * bsize, (i + 1) * bsize
            )
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print "usage: %s <data_json> <comparison_set> <low k> <high k>" % sys.argv[0]
        sys.exit(1)

    jsonbody = io.open(sys.argv[1], encoding='utf-8-sig')
    data = json.loads(jsonbody.read())

    comp_set = sys.argv[2]
    ks = list(range(int(sys.argv[3]), int(sys.argv[4]) + 1))

    do_train(data);
  
