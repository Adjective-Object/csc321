from numpy import *
import os, sys, io, json, face_utils
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

import pickle

current_generation = 0
rates = []

from myalexnet import *

def train(data, passes=100, bsize=1, snapshot_frequency=100):
    print "loading images"
    
    all_test = face_utils.load_fileset_multichannel(data["training"], "test",       0, None, 277 ** 2)
    for (name, _), i, o in zip(all_test[0], all_test[1], all_test[2]):
        argo = np.argmax(sess.run(out, feed_dict={x: i.reshape(1,277,277,3), y: o.reshape(1,6)}));
        print name, data["training"][argo]["name"]
        


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "usage: %s <data_json>" % sys.argv[0]
        sys.exit(1)

    jsonbody = io.open(sys.argv[1], encoding='utf-8-sig')
    data = json.loads(jsonbody.read())

    train(data);
