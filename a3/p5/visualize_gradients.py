import numpy
from matplotlib.image import imsave
from matplotlib import cm
import pickle

import os, sys

from myalexnet import *

#TODO: modify this
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "usage: %s <networks output file> <output_directory>" % sys.argv[0]

    snapshot = pickle.load(open(sys.argv[1]))
    for key in net_data.keys():
        pass

    # for n in range(hidden_weights.shape[1]):
    #     img = hidden_weights[:, n].reshape((32, 32))

   ##       imsave(
    #         os.path.join(sys.argv[2], "%04d.png" % n),
    #         img, 
    #         cmap=cm.coolwarm,
    #         format="png")
