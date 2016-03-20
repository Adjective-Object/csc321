import numpy
from matplotlib.image import *
from matplotlib import cm
import pickle
import tensorflow as tf

import os, sys

from myalexnet import *

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print "usage: %s <image file> <output directory>" % sys.argv[0]
    
    img = imread(sys.argv[1])
    img_in = img.reshape(1, 277,277, 3)

    radcliffe = np.array([[np.float32(0) for i in range(6)]])
    radcliffe[0][int(sys.argv[2])] = np.float32(1)

    dout_dimg = tf.gradients(NLL, x) 

    gradient = sess.run(dout_dimg, feed_dict = {
        x: img_in,
        y: radcliffe
    })[0]

    print "min:", np.min(gradient), "max:", np.max(gradient)
    gradient = (gradient - np.min(gradient)) / (np.max(gradient) - np.min(gradient)) * 255
    print "min:", np.min(gradient), "max:", np.max(gradient)

    imsave(
        os.path.join(sys.argv[3]),
        gradient.reshape(277,277,3),
        cmap=cm.ocean,
        format="png")
