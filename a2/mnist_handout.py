from pylab import *
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
from scipy.spatial.distance import euclidean

import cPickle

import os
from scipy.io import loadmat
import sys

def write(*xs):
    for x in xs:
        sys.stdout.write(str(x) if type(x) != str else x)
        sys.stdout.write(" ")


# constant characteristics of MNIST data
MNIST_INPUT_VECTOR_SIZE = 28 * 28 # 28x28 images
MNIST_NUM_CLASSES = 10            # digits 0 .. 9

def softmax(y):
    '''Return the output of the softmax function for the
    matrix of output y. y is an NxM matrix where N is the
    number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

def cost(y, y_):
    return -sum(y_*log(y)) 


#
# PART 2 : outputs as linear combination of inputs 
#


def generalize_input_data(inputs):
    const_inputs = empty((inputs.shape[0], 1))
    const_inputs.fill(1)
    return concatenate((const_inputs, inputs), axis=1)


def generalize_biases(weights, biases):
    ''' weights:I x O matrix
        inputs: n x I matrix
        biases: O x 1 matrix
    '''
    return concatenate((biases.reshape((1, weights.shape[1])), weights), axis=0)


def degeneralize_biases(combined):
    return (combined[0,:], combined[1:,:])


def linear_layer(inputs, weights):
    ''' 
        input_dim: I
        output_dim: O

        Inputs a n * I matrix of inputs
        weights a I * O matrix of weights of each of the inputs
        offsets a I * O matrix of weights of each of the inputs

        returns a n*O matrix of output matricies
    '''
    return dot(inputs, weights)


def part2(dataset):
    ''' dataset is the MNIST digit data
    '''

    # initialize all weights and offsets to some arbitrary values
    weights = empty((MNIST_INPUT_VECTOR_SIZE, MNIST_NUM_CLASSES))
    weights.fill(0.8 / MNIST_INPUT_VECTOR_SIZE)
    offsets = empty((MNIST_NUM_CLASSES))
    offsets.fill(0.2 / MNIST_NUM_CLASSES)

    load_samples(dataset, "test", 0, 10)

    # push through the linear layer
    g_wt = generalize_biases(weights, offsets)
    g_in = generalize_input_data(weights, offsets)
    outputs = linear_layer(g_in, g_wt)
    argmaxes = argmax(outputs, axis=1)

    # if the shape mismatches, panic
    if argmaxes.shape != expected_argmax.shape:
        print "mismatched shape : critical error!"
        print expected_argmax
        print argmaxes
        exit(1)

    # create array of bools that is true when 
    # the difference between expected & real is 0
    matches = expected_argmax - argmaxes
    matches_bool = invert(matches.astype(bool))

    return matches_bool




#
# PART 3 : gradient of the cost function
#

def neg_log_liklihood(target, probabilities):
    ''' Something something calculate log liklihood
    '''
    return cost(target, probabilities)

def grad_neg_log_liklihood(inp, weights, targets):
    ''' 
        inp: n * I array
        weights: I x O array
        targets: n * O array
    '''
    N, _ = inp.shape
    I, O = weights.shape

    output = empty((N, O))
    for n in range(N):
        output[n] = linear_layer(inp[n], weights)
 
    # print "output:", output.shape
    probability = softmax(output)
    # print "probability:", probability.shape
    pdiffs = (probability - targets)
    # print pdiffs.shape
    pdiffs = tile(pdiffs.reshape((N, 1, O)), (1, I, 1))
    # print pdiffs.shape
    inp_expanded = inp.reshape((N, I, 1))
    inp_expanded = tile(inp_expanded, (1, 1, O))

    gradient = pdiffs * inp_expanded

    return gradient

def approx_grad_neg_log_lklihood(inp, weights, targets, step=0.005):
    ''' 
        inp: n * I array
        weights: I x O array
        targets: n * O array
    '''
    N, _ = inp.shape
    I, O = weights.shape

    # print inp.shape, weights.shape

    output = linear_layer(inp, weights)
    # print output.shape
    # print targets.shape

    gradient = empty((N, I, O))
    for n in range(N):
        print "n = %d/%d" % (n, N)
        cur_input = inp[n, :].reshape((1, I))
        for o in range(O):
            for i in range(I):
                new_weights = weights.copy()

                new_weights[i,o] = weights[i,o] - step
                cost_right = neg_log_liklihood(
                    targets[n,:].flatten(),
                    softmax(linear_layer(cur_input, new_weights)).flatten())

                new_weights[i,o] = weights[i,o] + step
                cost_left = neg_log_liklihood(
                    targets[n,:].flatten(),
                    softmax(linear_layer(cur_input, new_weights)).flatten())

                # print "right", cost_right
                # print "left", cost_left
                # print "center", cost_center

                #print inp[n, :].shape, new_weights.shape, cost_left.shape, cost_right.shape

                # approx gradient by averaging left & right values
                gradient[n,i,o] = ((cost_right - cost_left) / (step * 2))

    return gradient









def part3(dataset):
    # push through the linear layer
    #outputs = grad_neg_log_liklihood(
    #    inputs, weights, offsets, expected_output)

    fakein1 = array([
        [0.4, 0.1, 0.3, 0.4],
        [0.4, 0.1, 0.3, 0.4],
        ])

    fakeWeights = array([
        [0.1, 0.1, 0.2, 0.9],
        [0.1, 0.5, 0.2, 0.2],
        ])

    fake_expected_output = array([
        [0.0, 1.0],
        [1.0, 0.0]
        ])

    fake_biases = array([0, 0])

    g_in = generalize_input_data(inputs) 
    g_wt = generalize_biases(fakeWeights, fake_biases)
    grad_neg_log_liklihood(
        g_in, g_wt, fake_expected_output)


def load_samples(dataset, name, batch_start, batch_end):
    # generate input matrix from all testing matricies
    # input matrix is a [n * input] matrix of inputs
    inputs = empty((0, MNIST_INPUT_VECTOR_SIZE))
    expected_output = empty((0, MNIST_NUM_CLASSES))
    numsamples = 20
    for i in range(0, 10):
        clazz = "%s%s" % (name, i)
        inputs = concatenate((inputs, dataset[clazz][batch_start:batch_end]))

        output_this = zeros((1, MNIST_NUM_CLASSES))
        output_this[0, i] = 1
        expected_output = concatenate(
            (expected_output,
            tile(output_this, (len(dataset[clazz][batch_start:batch_end]), 1) )))

    inputs = inputs/255.0
    return inputs, expected_output
    
def part4(dataset):
    
    # initialize all weights and offsets to some arbitrary values
    weights = np.random.random((MNIST_INPUT_VECTOR_SIZE, MNIST_NUM_CLASSES)) * 0.8/MNIST_INPUT_VECTOR_SIZE
    offsets = np.random.random((MNIST_NUM_CLASSES)) * 0.2 / MNIST_NUM_CLASSES

    inputs, expected_output = load_samples(dataset, "train", 0, 10)
    
    g_in = generalize_input_data(inputs) 
    g_wt = generalize_biases(weights, offsets)

    print "grad"
    gradients = grad_neg_log_liklihood(g_in, g_wt, expected_output)
    print "approx grad"
    approx_gradients = approx_grad_neg_log_lklihood(g_in, g_wt, expected_output)

    diffs = (approx_gradients - gradients).flatten()

    fig = plt.figure()
    plt.hist(diffs, 50)
    plt.savefig("difference_histogram.png", format="png")

    # remove outliers and ake another
    diffs_no_outliers = diffs[abs(diffs - np.mean(diffs)) < 2 * np.std(diffs)]
    outliers = np.setdiff1d(diffs, diffs_no_outliers)

    print diffs_no_outliers
    print outliers

    print outliers.size, diffs_no_outliers.size, diffs.size

    fig = plt.figure()
    plt.hist(diffs_no_outliers, 50)
    plt.savefig("difference_histogram_no_outliers.png", format="png")



#
# PART 5
#


def part5(dataset):
    # initialize weights
    weights = np.random.random((MNIST_INPUT_VECTOR_SIZE, MNIST_NUM_CLASSES)) * 0.8/MNIST_INPUT_VECTOR_SIZE
    offsets = np.random.random((MNIST_NUM_CLASSES)) * 0.2 / MNIST_NUM_CLASSES

    g_wt = generalize_biases(weights, offsets)

    # train on samples
    for lower, higher in zip(range(0,1001,50), range(50,1001,50)):
        print "training batch %d - %d", lower, higher
        # pull a batch out
        this_batch, this_batch_out = load_samples(dataset, "train", lower, higher)
        g_in = generalize_input_data(this_batch)

        #calculate the gradients for all of the inputs        
        gradients = grad_neg_log_liklihood(g_in, g_wt, this_batch_out)
        average_gradient = np.mean(gradients, axis=0)

        # go down the direction of the average gradient
        g_wt -= average_gradient * 0.01

    print "saving trained array.."
    np.save("part5", g_wt)

    print "testing array.."
    part5_test(dataset, g_wt)


def part5_test(dataset, g_wt):
    # g_in = np.load("part5.arr") 
    rate = np.empty((1000))
    data, real_outputs = load_samples(dataset, "test", 0, 1000)
    data = generalize_input_data(data)

    real_outputs = argmax(real_outputs, axis=1)
    outputs = argmax(softmax(linear_layer(data, g_wt)), axis=1)

    succCases = (real_outputs == outputs)
    rate = mean(succCases.astype(float))

    print "success rate", rate


#
# Part 6
# displaying as image
#

def part6():
    weights = np.load("part5.npy")
    biases, weights = degeneralize_biases(weights)

    # sum the weights of the pixels 
    for i in range(weights.shape[1]):
        fname = "./part6_%s.png" % i
        print "saving", fname

        this_digit = weights[:, i].reshape((28,28))

        fig = plt.figure()
        imshow(this_digit, cmap=cm.bwr)
        plt.savefig(fname, format='png')

#
# Part 7
# Multilayer Neural Network
#

def tanh_layer(y, (W, b)):    
    '''Return the output of a tanh layer for the input matrix y.
    y is an NxM matrix where N is the number of inputs for 
    a single case, and M is the number of cases'''
    print "tanh", y.shape, W.shape
    print "..", dot(y, W)
    return tanh(dot(y, W) + tile(b, (1, y.shape[0])).T)

def forward(x, (W0, b0), (W1, b1)):
    ''' Wx : weights of input
        bx : constant offsets of input

        returns the result of running input x through layers
        (W0, b0) and (W1, b1)

        x is an Nx? matrix where N is the # of cases...
    '''
    L0 = tanh_layer(x, (W0, b0))
    L1 = tanh_layer(L0, (W1, b1))
    output = softmax(L1)
    return L0, L1, output

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of 
    the cross-entropy cost function w.r.t the parameters of a 
    neural network

    TODO finsh
    '''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, ((1- L1**2)*dCdL1).T )

def approx_deriv_multilayer(
        (W0, b0), (W1, b1), inp, real_output, step=0.01):
    ''' 
        W0, b0, W1, b1: 
            weights and biases of the network

            I: input vector size
            H: hidden layer size
            O: output vector size
            N: number of cases

            W0 = I x H
            b0 = 1 x H

            W1 = H x O
            b1 = 1 x O

        input: N x I matix of inputs
        real_outputs: nxO matrix of outputs
    '''
        
    # get output of a normal run
    print inp
    L0, _, output = forward(inp, (W0, b0), (W1, b1))

    print output

    for n in range(inp.shape[0]):
        print inp[n].reshape(1, inp.shape[1])

        _, _, outpoot = forward(
            inp[n].reshape(1, inp.shape[1]),
            (W0, b0), (W1, b1))

        print outpoot

    # augment the matricies (combine the biases with the weight matricies)

    augmented_input = generalize_input_data(inp)
    augmented_interlayer_input = generalize_input_data(L0)

    augmented_layer0 = generalize_biases(W0, b0)
    augmented_layer1 = generalize_biases(W1, b1)

    # get augmented shapes

    H, O = augmented_layer1.shape
    N, I = augmented_input.shape

    print "O:", O, "N:", N, "I:", I, "H:", H

    gradient_layer0 = empty(augmented_layer0.shape)
    gradient_layer1 = empty(augmented_layer1.shape)

    print "input layer -> hidden layer"
    for h in range(H):
        for i in range(I):
            cost_left, cost_right = finite_difference_forward(
                augmented_input,    # input
                augmented_layer0,   # matrix to mutate
                augmented_layer1,   # matrix to forward through
                h, i,               # indicies to mutate
                real_output         # output to compare to
                )

            gradient_layer0[h, i] = (cost_right - cost_left) / (step * 2)


    print "hidden layer -> output layer"
    for o in range(O):
        for h in range(H):
            cost_left, cost_right = finite_difference(
                augmented_interlayer_input,    # input
                augmented_layer1,              # matrix to mutate
                o, h,                          # indicies to mutate
                real_output                    # output to compare to
                )

            gradient_layer1[o, h] = (cost_right - cost_left) / (step * 2)


    return (gradient_W0,
            gradient_b0,
            gradient_W1,
            gradient_b1)

def part7(dataset, snapshot):
    # # initialize all weights and offsets to some arbitrary values
    # W0 = snapshot["W0"]
    # b0 = snapshot["b0"].reshape((300,1))
    # W1 = snapshot["W1"]
    # b1 = snapshot["b1"].reshape((10,1))
    
    # trains, real_outputs = load_samples(dataset, "train", 0, 1)



    # approximations = approx_deriv_multilayer(
    #     (W0, b0), (W1, b1), trains, real_outputs)

    # 4 cases of 2 inputs
    inputs = array([
        [1,1],
        [2,2],
        [3,3],
        [4,4],
        ])
    print inputs.shape

    # broadcast 2 inputs to 3 hidden
    W0 = array([
        [0.1, 0.2, 0.3],
        [0.7, 0.2, 0.4],
        ])
    print W0.shape

    b0 = array([
        [0.1],
        [0.2],
        [0.3],
        ])

    # broadcast 3 hidden to 4 outputs
    W1 = array([
        [0.22, 0.33, 0.11, 0.01],
        [0.77, 0.22, 0.44, 0.01],
        [0.11, 0.22, 0.33, 0.01],
        ])

    b1 = array([
        [0.4],
        [0.3], 
        [0.2], 
        [0.1],
        ])

    real_outputs = array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        ])

    approx_deriv_multilayer(
        (W0, b0), (W1, b1),
        inputs, real_outputs
        )


def main():
    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    #Display the 150-th "5" digit from the training set
    # fig = plt.figure()
    # imshow(M["train5"][150].reshape((28,28)), cmap=cm.gray)
    # plt.savefig("./out.png", format='png')

    # #Load sample weights for the multilayer neural network
    snapshot = cPickle.load(open("snapshot50.pkl"))
    # W0 = snapshot["W0"]
    # b0 = snapshot["b0"].reshape((300,1))
    # W1 = snapshot["W1"]
    # b1 = snapshot["b1"].reshape((10,1))

    # #Load one example from the training set, and run it through the
    # #neural network
    # x = M["train5"][148:149].T
    # L0, L1, output = forward(x, (W0, b0), (W1, b1))
    # #get the index at which the output is the largest
    # y = argmax(output)

    part7(M, snapshot)

    ##################################################################
    # Code for displaying a feature from the weight matrix mW
    # fig = figure(1)
    # ax = fig.gca()    
    # heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
    # fig.colorbar(heatmap, shrink = 0.5, aspect=5)
    # show()
    ##################################################################



if __name__ == "__main__":
    main()