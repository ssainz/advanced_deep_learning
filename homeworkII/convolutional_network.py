from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np


# Define one layer of CNN:

x = tf.placeholder(tf.float32, [None, 784])
y_hat = tf.placeholder(tf.float32, [None, 10])

# The actual image to enter the convolution.
# First dim is the batch size.
# Second dim and third dim is the image.
# Fourth dim is the features size.
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Create the weights for the filters.
# The first two dimensions are the size of the 2-D filter.
# The third dimension is the number of channels in the image (say 3 if it is RGB).
# The fourth dimension is the output features (how many features).
W_conv1 = tf.Variable(tf.random_uniform([3, 3, 1, 16], minval=0.0, maxval=0.3))
# The bias equal to the number of output features.
b_conv1 = tf.Variable(tf.random_uniform([16], minval=0.0, maxval=0.3))
# Convolution. The strides is one number per dimension:
#    1st dimension is the batch size, second and third dimension is the image, fourth dim is the features
# The padding is "VALID" which means no padding.
# Or, "SAME" which is explained here: https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
# "SAME" padding means that the size of the image after convolution will be the same as the image itself!!
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='VALID')  + b_conv1 )
# Max pool. For both the ksize (window size of the max pool, and the strides:
#  The first dimension is the batch number , 2nd and 3rd dims are the image, 4th dimension is feature number.
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


W_conv2 = tf.Variable(tf.random_uniform([5, 5, 16, 32], minval=0.0, maxval=0.3))
b_conv2 = tf.Variable(tf.random_uniform([32], minval=0.0, maxval=0.3))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# 5x5
# flatten
W_flatten = tf.Variable(tf.random_uniform([4*4*32, 1024], minval=0.0, maxval=0.3))
b_flatten = tf.Variable(tf.random_uniform([1024], minval=0.0, maxval=0.3))
flatten_pool = tf.reshape(h_pool2, [-1,4*4*32])

# apply fully connected layer of 1024 neurons
fc7 = tf.nn.relu(tf.matmul(flatten_pool, W_flatten) + b_flatten)

# Output layer:
W_output = tf.Variable(tf.random_uniform([1024, 10], minval=0.0, maxval=0.3))
b_output = tf.Variable(tf.random_uniform([10], minval=0.0, maxval=0.3))

# Apply output layer
y = tf.nn.relu(tf.matmul(fc7, W_output) + b_output)

# softmax to the final 10 classes (mnist)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_hat, logits=y)

loss = tf.reduce_mean(cross_entropy)

# Optimizer:
# learning rate:
eta = 0.01
train = tf.train.AdamOptimizer(eta).minimize(loss)

# Start training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# Load data:
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
for i in range(6000):
    X_raw, Y_raw = mnist.train.next_batch(10)
    #Y_raw = np.reshape(Y_raw, (-1,10))
    [cur_train, cur_loss] =  sess.run([train,loss], {x:X_raw, y_hat: Y_raw})
    print("Iteration (%s), loss (%s)"%(i, cur_loss))


#print("Len of X (%s)" % len(X) )
#print("Len of Y (%s)" % len(Y) )
#print("Len of X[0] (%s)" % len(X[0]) )
#print("Len of Y[0] (%s)" % len(Y[0]) )