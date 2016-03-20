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
    
    img = imread('861.png')
    img_in = img.reshape(1, 277,277, 3)
    one_hot = np.array([[np.float32(0) for i in range(6)]])
    one_hot[0][1] = np.float32(1)
    p_tensor = sess.run(prob, feed_dict={
                    x: img_in,
                    y: one_hot
                })
    #this is the problem area
    dout_dimg = tf.gradients(p_tensor, tf.reshape(x, [-1]))

    imsave(
        os.path.join(sys.argv[2], "radcliffe_grad.png"),
        dout_dimg, 
        cmap=cm.ocean,
        format="png")