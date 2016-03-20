'''
based on Aymeric Damien's MNIST tensorflow implementation from the 
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf



######################
# Network Parameters #
######################

img_size = 64 # data input (img shape: 64*64)
n_input = img_size ** 2 * 3
n_classes = 6 # total classes (3 actors, 3 actresses)

# tf Graph input
network_input = tf.placeholder(tf.float32, [None, n_input])
network_expected = tf.placeholder(tf.float32, [None, n_classes])

# initialize weights and biases
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([4*256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}



####################
# helper functions #
####################

def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)



#####################
# ALEXNET STRUCTURE #
#####################

# Reshape input picture
x = tf.reshape(network_input, shape=[-1, img_size, img_size, 1])

# Convolution Layer1
conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])
# Max Pooling (down-sampling)
pool1 = max_pool('pool1', conv1, k=2)
# Apply Normalization
norm1 = norm('norm1', pool1, lsize=4)


# Convolution Layer2
conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
# Max Pooling (down-sampling)
pool2 = max_pool('pool2', conv2, k=2)
# Apply Normalization
norm2 = norm('norm2', pool2, lsize=4)


# Convolution Layer3
conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
# Max Pooling (down-sampling)
pool3 = max_pool('pool3', conv3, k=2)
# Apply Normalization
norm3 = norm('norm3', pool3, lsize=4)


# Fully connected layer
dense1 = tf.reshape(norm3, [-1, weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
dense1 = tf.nn.relu(tf.matmul(dense1, weights['wd1']) + biases['bd1'], name='fc1') # Relu activation
dense2 = tf.nn.relu(tf.matmul(dense1, weights['wd2']) + biases['bd2'], name='fc2') # Relu activation

# Output, class prediction
out = tf.matmul(dense2, weights['out']) + biases['out']
prediction = out



###########
# OUTPUTS #
###########

# declare the cost function (negative log likelihood), training step
NLL = -tf.reduce_sum(network_expected * tf.log(prediction))
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(NLL)

# Evaluate model
num_correct = tf.equal(
    tf.argmax(prediction, 1),
    tf.argmax(network_expected,1))
accuracy = tf.reduce_mean(
    tf.cast(num_correct, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()



###############################
# Finally, Evaluate the thing #
###############################

sess = tf.Session()
sess.run(init)

