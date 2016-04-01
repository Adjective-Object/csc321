import numpy as np
from matplotlib import pyplot as plt
import random
import sys

np.random.seed(555)
random.seed(555)


def color_from_activation(weight):
    weight = float(weight)

    weight = max(-1, min(1, weight))

    low = [0,0,255]
    mid = [255,255,255]
    high = [255, 0, 0]

    mix = ([weight * h + (1 - weight) * m for m,h in zip(mid, high)]
            if weight > 0 
            else [-weight * l + (1 + weight) * m for l,m in zip(low, mid)])

    return "#%02x%02x%02x" % tuple(mix)

def visualize_sample(h, neuron, seed_ix, n):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """

    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in xrange(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)


        # print as html
        char = ix_to_char[ix]
        if char == '\n':
            char = '&nbsp;<br/>'

        color = color_from_activation(h[neuron] / (h.max() - h.min()))
        sys.stdout.write(
            "<span style='background-color: %s;'>%s</span>" % 
            (color, char))

    return ixes



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
    
    initial_char = random.choice(chars)
    memory_state = np.zeros(bh.shape)
    visualize_sample(memory_state, int(sys.argv[1]), char_to_ix[initial_char], 2000)



