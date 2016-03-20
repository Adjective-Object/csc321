import numpy
from matplotlib.image import *
from matplotlib import cm
import pickle
import tensorflow as tf

import os, sys

from myalexnet import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "usage: %s <image file>" % sys.argv[0]
    
    img = imread(sys.argv[1])
    img_in = img.reshape(1, 277,277, 3)
    one_hot = np.array([[np.float32(0) for i in range(6)]])
    one_hot[0][1] = np.float32(1)
    out_tensor = sess.run(out, feed_dict={
                    x: img_in,
                    y: one_hot
                })
    print tf.gradients(out, x)
