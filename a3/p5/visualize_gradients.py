import numpy
from matplotlib.image import imsave
from matplotlib import cm
import pickle

import os, sys

from myalexnet import *

#TODO: modify this
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "usage: %s <image file>" % sys.argv[0]

    
