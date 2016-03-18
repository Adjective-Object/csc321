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

from random import randint

import cPickle

import os
from scipy.io import loadmat
import sys

def write(*xs):
    for x in xs:
        sys.stdout.write(str(x) if type(x) != str else x)
        sys.stdout.write(" ")

    sys.stdout.flush()

def dtanh(y):
    return 1.0 - (y** 2)



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
    ''' inputs: I x n matrix
    '''
    const_inputs = empty((inputs.shape[0], 1))
    const_inputs.fill(1)
    return concatenate((const_inputs, inputs), axis=1)


def generalize_biases(weights, biases):
    ''' weights:I x O matrix
        biases: O x 1 matrix
    '''
    return concatenate((biases.reshape((1, weights.shape[1])), weights), axis=0)


def degeneralize_biases(combined):
    return (combined[0,:], combined[1:,:])


def linear_layer(inputs, weights):
    '''
        input_dim: I
        output_dim: O

        Inputs a  I * n matrix of inputs
        weights a I * O matrix of weights of each of the inputs

        returns a n*O matrix of output matricies

        inputs should be generalized s.t. biases are the first index of the
        thing
    '''

    return dot(inputs, weights)

def part1(dataset):
    for i in range(0,10):
        for j in range(0,10):
            imsave(
                "writeup/part1/%d_%d.png"%(i,j),
                dataset["test%d"%i][j].reshape((28,28)),
                format="png", cmap=cm.gray)



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
# Gradient of the cost
#

def neg_log_likelihood(target, probabilities):
    ''' Something something calculate log likelihood
    '''
    return cost(probabilities.T, target.T)

def grad_neg_log_likelihood(inp, weights, targets):
    N, _ = inp.shape
    I, O = weights.shape

    output = linear_layer(inp.reshape((N, I)), weights)

    probability = softmax(output)
    pdiffs = (probability - targets)
    pdiffs = tile(pdiffs.reshape((N, 1, O)), (1, I, 1))
    inp_expanded = inp.reshape((N, I, 1))
    inp_expanded = tile(inp_expanded, (1, 1, O))

    return pdiffs * inp_expanded

def approx_grad_neg_log_likelihood(inp, weights, targets, step=0.005):
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
    for o in range(O):
        for i in range(I):
            new_weights = weights.copy()

            new_weights[i,o] = weights[i,o] - step
            cost_right = neg_log_likelihood(
                targets,
                softmax(linear_layer(inp, new_weights)))

            new_weights[i,o] = weights[i,o] + step
            cost_left = neg_log_likelihood(
                targets,
                softmax(linear_layer(inp, new_weights)))

            # print "right", cost_right
            # print "left", cost_left
            # print "center", cost_center

            #print inp[n, :].shape, new_weights.shape, cost_left.shape, cost_right.shape

            # approx gradient by averaging left & right values
            gradient[:,i,o] = ((cost_right - cost_left) / (step * 2))

    return gradient

def part3(dataset):
    # push through the linear layer
    #outputs = grad_neg_log_likelihood(
    #    inputs, weights, offsets, expected_output)

    inputs = array([
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
    grad_neg_log_likelihood(
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





#
# Part4 - approximating the gradient
# Yeah
#

def part4(dataset):

    np.random.seed(1)

    # initialize all weights and offsets to some arbitrary values
    weights = np.random.random((MNIST_INPUT_VECTOR_SIZE, MNIST_NUM_CLASSES)) * 0.8/MNIST_INPUT_VECTOR_SIZE
    offsets = np.random.random((MNIST_NUM_CLASSES)) * 0.2 / MNIST_NUM_CLASSES

    inputs, expected_output = load_samples(dataset, "train", 0, 100)

    g_in = generalize_input_data(inputs)
    g_wt = generalize_biases(weights, offsets)

    print "grad"
    gradients = grad_neg_log_likelihood(g_in, g_wt, expected_output)
    print "approx grad"
    approx_gradients = approx_grad_neg_log_likelihood(g_in, g_wt, expected_output)

    diffs = (approx_gradients - gradients).flatten()

    fig = plt.figure()
    plt.hist(diffs, 50)
    plt.title("approx vs actual gradient (single layer)")
    plt.savefig("writeup/part4/difference_histogram.png", format="png")

    # remove outliers and ake another
    diffs_no_outliers = diffs[abs(diffs - np.mean(diffs)) < 2 * np.std(diffs)]
    outliers = np.setdiff1d(diffs, diffs_no_outliers)

    print diffs_no_outliers
    print outliers

    print outliers.size, diffs_no_outliers.size, diffs.size

    fig = plt.figure()
    plt.hist(diffs_no_outliers, 50)
    plt.title("approx vs actual gradient (single layer, no outliers)")
    plt.savefig("writeup/part4/difference_histogram_no_outliers.png", format="png")


#
# PART 5
# Training the single-layer neural network on the
# input data and showing it
#


def part5(dataset):
    # initialize weights
    weights = np.random.random((MNIST_INPUT_VECTOR_SIZE, MNIST_NUM_CLASSES)) * 0.8/MNIST_INPUT_VECTOR_SIZE
    offsets = np.random.random((MNIST_NUM_CLASSES)) * 0.2 / MNIST_NUM_CLASSES

    g_wt = generalize_biases(weights, offsets)

    test_results = []
    # train on samples
    batchsize = 50
    for lower, higher in zip(range(0,1001,batchsize), range(batchsize,1001,batchsize)):
        print "training batch %d to %d of 1000" % (lower, higher)
        # pull a batch out
        this_batch, this_batch_out = load_samples(dataset, "train", lower, higher)
        g_in = generalize_input_data(this_batch)

        #calculate the gradients for all of the inputs
        gradients = grad_neg_log_likelihood(g_in, g_wt, this_batch_out)
        average_gradient = np.mean(gradients, axis=0)

        test_results.append(part5_test(dataset, g_wt))

        print "success rate", test_results[-1]

        # go down the direction of the average gradient
        g_wt -= average_gradient * 0.01


    print "saving trained array.."
    np.save("part5", g_wt)
    test_results.append(part5_test(dataset, g_wt))

    plot_sample_part5(test_results, dataset, g_wt)



def plot_sample_part5(test_results, dataset, g_wt):
    # plotting
    make_digit_sample(dataset, g_wt)
    test_results = array(test_results).T

    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.set_title("classification rates for all digits")

    for i in range(11):
        ax.plot(test_results[i])

    ax.set_ylim(top=1.5)

    ax.legend(["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], fontsize=6)


    ax2 = fig.add_subplot(2,1,2)
    ax2.set_title("average classification rate")

    ax2.plot(test_results[0])

    ax2.set_ylim(top=1.5)

    plt.savefig("learning_rate.png", format='png')


def make_digit_sample(dataset, g_wt):
    raw_data, expected_outputs = load_samples(dataset, "test", 0, 1000)
    data = generalize_input_data(raw_data)

    expected_outputs = argmax(expected_outputs, axis=1)
    outputs = argmax(softmax(linear_layer(data, g_wt)), axis=1)

    print expected_outputs.shape, outputs.shape, raw_data.shape


    num_succ = 0
    num_failures = 0
    elems = list(range(outputs.size))
    random.shuffle(elems)
    for i in elems:
        expected = expected_outputs[i]
        calculated = outputs[i]

        case = raw_data[i]

        if expected != calculated and num_failures < 10:
            imsave("writeup/part5/failure_%d.png" % num_failures, case.reshape((28,28)), cmap=cm.gray)
            num_failures+=1

        if expected == calculated and num_succ < 20:
            imsave("writeup/part5/succ_%d.png" % num_succ, case.reshape((28,28)), cmap=cm.gray)
            num_succ+=1
    print num_succ, num_failures



def part5_test(dataset, g_wt):
    # g_in = np.load("part5.arr")
    data, expected_outputs = load_samples(dataset, "test", 0, 1000)
    data = generalize_input_data(data)

    expected_outputs = argmax(expected_outputs, axis=1)
    outputs = argmax(softmax(linear_layer(data, g_wt)), axis=1)

    succCases = (expected_outputs == outputs)
    rates = [mean(succCases[lower:upper].astype(float))
                for (lower, upper) in
                zip(range(0,1001,100), range(100,1001,100)) ]
    rate_all = sum(rates)/len(rates)

    return [rate_all] + rates







#
# Part 6
# displaying the input weights as an image
#

def part6():
    weights = np.load("part5.npy")
    biases, weights = degeneralize_biases(weights)

    # sum the weights of the pixels
    for i in range(weights.shape[1]):
        fname = "writeup/part6/part6_%s.png" % i
        print "saving", fname

        this_digit = weights[:, i].reshape((28,28))
        imsave(fname, this_digit, cmap=cm.coolwarm)





#
# Part 7
# Multilayer Neural Network
#

def _tanh_layer(y, (W, b)):
    '''Return the output of a tanh layer for the input matrix y.
    y is an NxM matrix where N is the number of inputs for
    a single case, and M is the number of cases'''
    #print "tanh", W.T.shape, y.shape
    #print "..", dot(W.T, y)
    return tanh(dot(W.T, y)+b)


def forward(x, (W0, b0), (W1, b1)):
    ''' Wx : weights of input
        bx : constant offsets of input

        returns the result of running input x through layers
        (W0, b0) and (W1, b1)

    '''
    L0 = _tanh_layer(x, (W0, b0))
    L1 = _tanh_layer(L0, (W1, b1))
    output = softmax(L1)
    return L0, L1, output

def grad_multilayer((W0, b0), (W1, b1), inp, expected_output):
    '''Incomplete function for computing the gradient of
    the cross-entropy cost function w.r.t the parameters of a
    neural network

    TODO finsh

    inp:             N x I matrix of inputs
    expected_output: N x O matrix of outputs

    '''

    # convenience variables
    N, I = inp.shape
    H = W0.shape[1]
    O = W1.shape[1]

    # convert from N-first order to N-last order
    inp = inp.T
    b0 = b0.reshape((H, 1))
    b1 = b1.reshape((O, 1))
    expected_output = expected_output.T

    # run through the neural network
    L0, L1, prediction = forward(inp, (W0, b0), (W1,b1))

    # calculate the gradient
    dCdL1 = (prediction - expected_output)
    dL1dL0 = dtanh(W1)
    dCdL0 = dot(dL1dL0, dCdL1)

    dL1dW1 = dtanh(L0)  # L1 = tanh(dot(W1, L0))
    dL0dW0 = dtanh(inp) # L0 = tanh(dot(W0, inp)

    gradients_W1 = dCdL1.reshape((1, O, N)) * dL1dW1.reshape((H, 1, N))
    gradients_W0 = dCdL0.reshape((1, H, N)) * dL0dW0.reshape((I, 1, N))

    return (gradients_W0, dCdL0), (gradients_W1, dCdL1), prediction





#
# Part 8 : Approximating the gradient for a 2 layer
# Neural Network
#

def approx_grad_multilayer(
        (W0, b0), (W1, b1),
        inp, expected_output, step=0.01):
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
        expected_outputs: nxO matrix of outputs
    '''

    ###################################
    # transforming to internal format #
    ###################################

    N, I = inp.shape
    H, O = W1.shape

    inp = inp.T
    expected_output = expected_output.T
    b0 = b0.reshape((H, 1))
    b1 = b1.reshape((O, 1))

    ########################
    # Layers for NN & Grad #
    ########################

    L0, _, _ = forward(inp, (W0, b0), (W1, b1))

    print "O:", O, "N:", N, "I:", I, "H:", H

    gradient_layer0 =  empty((I, H, N))
    gradient_layer1 =  empty((H, O, N))
    gradient_b0 = empty((H, N))
    gradient_b1 = empty((O, N))

    print "hidden layer -> output layer"
    for o in range(O):
        numdone = int((o*1.0/O)*30)
        for h in range(H):
            W1_copy = copy(W1)
            W1_copy[h, o] += step
            cost_right = cost(
                softmax(_tanh_layer(L0, (W1_copy, b1))),
                expected_output)

            W1_copy = copy(W1)
            W1_copy[h, o] -= step
            cost_left = cost(
                softmax(_tanh_layer(L0, (W1_copy, b1))),
                expected_output)

            gradient_layer1[h, o, :] = (cost_right - cost_left) / (step * 2)

        b1_copy = copy(b1)
        b1_copy[o] += step
        _, _, probs = forward(inp, (W0, b0), (W1, b1_copy))
        cost_right = cost(probs, expected_output)

        b1_copy = copy(b1)
        b1_copy[o] -= step
        _, _, probs = forward(inp, (W0, b0), (W1, b1_copy))
        cost_left = cost(probs, expected_output)

        gradient_b1[o ,:] = (cost_right - cost_left) / (step * 2)

    print "\r["+"#"*30+"]", "%d/%d" %(O, O)

    print gradient_layer1.shape, gradient_layer1[:,0,:].shape
    # print transpose(gradient_layer1[:,1:,:], (1,2,0)).shape
    # print gradient_layer1[:,0,:].T.shape
    np.save("grad_W1_approx", gradient_layer1)
    np.save("grad_b1_approx", gradient_b1)


    print "input layer -> hidden layer"
    for h in range(H):
        numdone = int((h*1.0/H)*30)
        write("\r["+"#"*numdone +" "*(30-numdone)+"]", "%d/%d" %(h, H))
        for i in range(I):
            W0_copy = copy(W0)
            W0_copy[i, h] += step
            _, _, probs = forward(inp, (W0_copy, b0), (W1, b1))
            cost_right = cost(probs, expected_output)

            W0_copy = copy(W0)
            W0_copy[i, h] -= step
            _, _, probs = forward(inp, (W0_copy, b0), (W1, b1))
            cost_left = cost(probs, expected_output)

            # print i, I, h, H, gradient_layer0.shape
            gradient_layer0[i, h, :] = (cost_right - cost_left) / (step * 2)

        b0_copy = copy(b0)
        b0_copy[h] += step
        _, _, probs = forward(inp, (W0, b0_copy), (W1, b1))
        cost_right = cost(probs, expected_output)

        b0_copy = copy(b0)
        b0_copy[h] -= step
        _, _, probs = forward(inp, (W0, b0_copy), (W1, b1))
        cost_left = cost(probs, expected_output)

        gradient_b0[h, :] = (cost_right - cost_left) / (step * 2)

    print "\r["+"#"*30+"]", "%d/%d" %(H, H)

    np.save("grad_W0_approx", gradient_layer0)
    np.save("grad_b0_approx", gradient_b0)

def part8_gen_approx(dataset, snapshot, samps=100):
    # # initialize all weights and offsets to some arbitrary values
    W0 = snapshot["W0"]
    b0 = snapshot["b0"]
    W1 = snapshot["W1"]
    b1 = snapshot["b1"]

    inputs, expected_outputs = load_samples(dataset, "train", 0, samps)

    # # 4 cases of 2 inputs
    # inputs = array([
    #     [1,3,5,7],
    #     [2,4,6,8],
    #     ]).T

    # # broadcast 2 inputs to 3 hidden
    # W0 = array([
    #     [0.1, 0.2, 0.3],
    #     [0.1, 0.2, 0.3],
    #     ])
    # print W0.shape

    # b0 = array([
    #     [0.1],
    #     [0.2],
    #     [0.3],
    #     ])

    # # broadcast 3 hidden to 4 outputs
    # W1 = array([
    #     [0.22, 0.33, 0.11, 0.01],
    #     [0.77, 0.22, 0.44, 0.01],
    #     [0.11, 0.22, 0.33, 0.01],
    #     ])

    # b1 = array([
    #     [0.4],
    #     [0.3],
    #     [0.2],
    #     [0.1],
    #     ])

    # expected_outputs = array([
    #     [1,0,0,0],
    #     [0,1,0,0],
    #     [0,0,1,0],
    #     [0,0,0,1],
    #     ])

    # approximations = approx_deriv_multilayer(
    #     (W0, b0), (W1, b1), trains, expected_outputs)

    approx_grad_multilayer(
        (W0, b0), (W1, b1),
        inputs, expected_outputs
        )


def part8_gen_calc(dataset, snapshot, samps=100):
    # # initialize all weights and offsets to some arbitrary values
    W0 = snapshot["W0"]
    b0 = snapshot["b0"]
    W1 = snapshot["W1"]
    b1 = snapshot["b1"]

    inputs, expected_outputs = load_samples(dataset, "train", 0, samps)

    (grad_W0, grad_b0), (grad_W1, grad_b1), _ = grad_multilayer(
        (W0, b0), (W1, b1),
        inputs, expected_outputs
        )

    # save output
    np.save("grad_W0_calc", grad_W0)
    np.save("grad_b0_calc", grad_b0)
    np.save("grad_W1_calc", grad_W1)
    np.save("grad_b1_calc", grad_b1)


def part8_compare():

    W0_calc   = np.load("grad_W0_calc.npy")
    b0_calc   = np.load("grad_b0_calc.npy")
    W1_calc   = np.load("grad_W1_calc.npy")
    b1_calc   = np.load("grad_b1_calc.npy")

    W0_approx = np.load("grad_W0_approx.npy")
    b0_approx = np.load("grad_b0_approx.npy")
    W1_approx = np.load("grad_W1_approx.npy")
    b1_approx = np.load("grad_b1_approx.npy")

    for name, calc, approx in zip(
        ["W0",      "b0",      "W1",      "b1"      ],
        [W0_calc,   b0_calc,   W1_calc,   b1_calc   ],
        [W0_approx, b0_approx, W1_approx, b1_approx]):

        print name, "calc:", calc.shape, "approx:", approx.shape


    for name, calc, approx in zip(
        ["W0",      "b0",      "W1",      "b1"      ],
        [W0_calc,   b0_calc,   W1_calc,   b1_calc   ],
        [W0_approx, b0_approx, W1_approx, b1_approx]):

        approx -= calc

        fig = plt.figure()
        plt.hist(approx.flatten(), 50)
        print "writing difference_histogram_%s.png"%name
        plt.title("approximate vs calculated cost for values in %s" % name)
        plt.savefig("difference_histogram_%s.png"%name, format="png")



#
# Part 9
# Using the gradient
#

def part9(dataset, snapshot):

    np.random.seed(1)

    W0, b0 = snapshot["W0"], snapshot["b0"].reshape((300, 1))
    W1, b1 = snapshot["W1"], snapshot["b1"].reshape((10, 1))

    # print "dims", W0.shape, b0.shape, W1.shape, b1.shape

    batchsize = 5
    rates = []
    for lower, higher in zip(range(0,1001,batchsize), range(batchsize, 1001, batchsize)):
        print "training multilayer batch %d to %d of 1000" % (lower, higher)

        inputs, expected_outputs = load_samples(dataset, "train", lower, higher)
        (gW0, gb0), (gW1, gb1), outputs = grad_multilayer(
            (W0, b0), (W1, b1), inputs, expected_outputs)

        # print gW0.shape, gb0.shape, gW1.shape, gb1.shape
        gW0 = mean(gW0, axis=2)
        gb0 = mean(gb0, axis=1).reshape(300, 1)

        gW1 = mean(gW1, axis=2)
        gb1 = mean(gb1, axis=1).reshape(10, 1)

        # print "dims", gW0.shape, gb0.shape, gW1.shape, gb1.shape

        thisRate, _ = get_rates_part9(dataset, (W0, b0), (W1, b1))
        rates.append(thisRate)
        print "success rate: %s" % rates[-1]

        W0 -= gW0 * 0.01
        b0 -= gb0 * 0.01
        W1 -= gW1 * 0.01
        b1 -= gb1 * 0.01

    finalRate, matches = get_rates_part9(dataset, (W0, b0), (W1, b1))
    rates.append(finalRate)

    np.save("final_network_W0", W0)
    np.save("final_network_b0", b0)
    np.save("final_network_W1", W1)
    np.save("final_network_b1", b1)

    # draw the curve
    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)
    ax.set_title("learning curve for the multilayer neural network")
    ax.set_ylim(top=1.0)
    ax.plot(rates)

    plt.savefig("./multilayer_learning_curve.png", format='png')

    make_digit_sample_9(dataset, matches)


def get_rates_part9(dataset, (W0, b0), (W1, b1)):
    # g_in = np.load("part5.arr")
    inputs, expected_outputs = load_samples(dataset, "train", 0, 1000)
    # print inputs.shape, expected_outputs.shape
    # print W0.shape
    _, _, outputs = forward(inputs.T, (W0, b0), (W1, b1))
    # print outputs, expected_outputs
    expected_outputs = argmax(expected_outputs.T, axis=0)
    outputs = argmax(outputs, axis=0)
    #print expected_outputs.shape, outputs.shape

    succCases = expected_outputs - outputs
    #print outputs
    #print expected_outputs
    rate = 1.0 - (float(count_nonzero(succCases)) / succCases.size)

    return rate, succCases

def make_digit_sample_9(dataset, matches):
    inputs, _ = load_samples(dataset, "test", 0, 10000)

    num_failures = 0
    num_succ = 0
    while num_failures < 10 or num_succ < 20:
        i = randint(0, len(matches) - 1)

        case = inputs[i]

        if not matches[i] and num_failures < 10:
            imsave("writeup/part9/multilayer_failure_%d.png" % num_failures,
                case.reshape((28,28)),
                cmap=cm.gray)
            num_failures+=1

        if matches[i] and num_succ < 20:
            print "succ", num_succ
            imsave("writeup/part9/multilayer_succ_%d.png" % num_succ,
                case.reshape((28,28)),
                cmap=cm.gray)
            num_succ+=1



#
# Part 10
#

def part10():
    W0 = np.load("final_network_W0.npy")

    for n in range(W0.shape[1]):
        imsave("writeup/part10/%03d.png"%(n),
                W0[:, n].reshape((28,28)),
                format='png', cmap=cm.coolwarm)





def main():
    np.seterr('raise')

    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    #Display the 150-th "5" digit from the training set
    # fig = plt.figure()
    # imshow(M["train5"][150].reshape((28,28)), cmap=cm.gray)
    # plt.savefig("./out.png", format='png')

    # #Load sample weights for the multilayer neural network
    snapshot = cPickle.load(open("snapshot50.pkl"))
    W0 = snapshot["W0"]
    # b0 = snapshot["b0"].reshape((300,1))
    # W1 = snapshot["W1"]
    # b1 = snapshot["b1"].reshape((10,1))

    # #Load one example from the training set, and run it through the
    # #neural network
    # x = M["train5"][148:149].T
    # L0, L1, output = forward(x, (W0, b0), (W1, b1))
    # #get the index at which the output is the largest
    # y = argmax(output)

    # Part 1 - drawing the dataset
    part1(M)

    # Part 2 - calculating the network
    part2(M)

    # Part 3 - calculating the gradient
    part3(M)

    # Part 4 - approxmating the gradient
    part4(M)

    # Part 5 - Use 1 layer network
    part5(M)

    # Part 6 - Vidualize 1 layer network
    part6()

    # Part 7 - Computing the gradient
    part7(M)

    # Part 8 - Approximating the gradient for a multilayer network
    part8_gen_calc(M, snapshot, 10)
    part8_gen_approx(M, snapshot, 10)
    part8_compare()

    # Part 9 - training with the multilayer network
    part9(M, snapshot)

    # Part 10 - visualizing the 2nd layer of the multilayer network
    part10()

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
