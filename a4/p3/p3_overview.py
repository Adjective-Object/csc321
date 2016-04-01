import numpy as np
from matplotlib import pyplot as plt
import random

np.random.seed(555)
random.seed(555)

def normalize(arr):
   return ((arr - arr.min()) / (arr.max() - arr.min()) - 0.5) * 2

def inspect(arr, highlight_indexes, threshold=0.5):
    # create a graph with newline and space highlighted along border
    arr = normalize(arr)
    highlighted_numbers = []
    for x in range(arr.shape[1]):
        for ind in highlight_indexes:
            if abs(arr[ind,x]) > threshold:
                print "input %03d (weight %s)-> implies -> output %03d" % (x, arr[ind,x], ind)
                highlighted_numbers.append(x)

    arr_graph = np.pad(arr,
            [(1,0), (1,0)],
            mode='constant', constant_values=0)

    for ind in highlight_indexes:
        arr_graph[ind + 1, 0] = 1
   
    plt.imshow(arr_graph, cmap=plt.cm.coolwarm, interpolation='nearest')
    plt.show()

    return highlighted_numbers



if __name__ == "__main__":
    # load the file and unpak the weights
    snapshot = np.load(open("dataset/char-rnn-snapshot.npz"))
    Wxh = snapshot["Wxh"] 
    Whh = snapshot["Whh"]
    Why = snapshot["Why"]
    bh = snapshot["bh"]
    by = snapshot["by"]

    mWxh, mWhh, mWhy = snapshot["mWxh"], snapshot["mWhh"], snapshot["mWhy"]
    mbh, mby = snapshot["mbh"], snapshot["mby"]

    # get other parameters of the matrix
    chars, data_size, vocab_size, char_to_ix, ix_to_char = [
        snapshot[x].tolist() for x in [
            "chars", "data_size", "vocab_size", 
            "char_to_ix", "ix_to_char"
        ]]


    print
    print "Wxh (looking for things that ':' causes"
    colon_features = inspect(Wxh.T, [char_to_ix[':']])

    print
    print "Whh.T (looking for how the featues triggered by ':' are pushed through history"
    next_time_step_colon_features = inspect(Whh.T, colon_features)

    print
    print "Why (looking for features that directly cause newlines and spaces)' '"
    inds = inspect(Why, [char_to_ix['\n'], char_to_ix[' ']])

    print "features triggered by ':':             ", colon_features
    print "features trig by ':' on t+1:           ", next_time_step_colon_features
    print "things that cause newlines and spaces: ", inds



