import tensorflow as tf
import numpy as np
import random
import math
import matplotlib.pyplot as plt

sess = tf.Session()


def generate_data_sin(num_sequence, index, echo_step):
    #start = random.randint(0,101)
    #index = 0
    #eight_of_pi = math.pi / 8
    radian_step = math.pi / 16
    t = np.arange(index, num_sequence+index)
    xs = np.sin(t * radian_step)
    ys = np.roll(xs, -1 * echo_step)
    #ys = np.rollaxis(xs, echo_step)
    #ys[0:echo_step] = 0
    return xs, ys, num_sequence+index

num_of_recurrences = 40
size_of_input_x = 1

size_of_h_t = 10

size_of_output_y = 1

echo_step = 2

num_of_epochs = 5000

# first dimension are the input size of each x_i vector,
# second dimension is the t_i (how many recurrences),
# the third dimension is number of samples.
X = tf.placeholder(tf.float32, shape=(size_of_input_x,num_of_recurrences,1))
# first dimension is size of y,
# second dimension is how many recurrencies
# third dimension is how many samples
Y_hat = tf.placeholder(tf.float32, shape=(size_of_output_y,num_of_recurrences))



h_t = tf.Variable(tf.random_uniform([size_of_h_t,1], minval=0.0, maxval=0.03), name="h_t")
W_i = tf.Variable(tf.random_uniform([size_of_h_t, size_of_input_x], minval=0.0, maxval=0.03), name="W_i")
W_h = tf.Variable(tf.random_uniform([size_of_h_t,size_of_h_t], minval=0.0, maxval=0.03), name="W_h")
W_o = tf.Variable(tf.random_uniform([size_of_output_y,size_of_h_t], minval=0.0, maxval=0.03), name="W_o")

X_is = tf.unstack(X, axis=1)
Ys = []
for x_i in X_is:
    h_t = tf.sigmoid(tf.matmul(W_i, x_i) + tf.matmul(W_h, h_t))
    #y_i = tf.nn.softmax(tf.matmul(W_o,h_t))
    y_i = tf.matmul(W_o,h_t)
    Ys.append(y_i)

Ys_tensor = tf.convert_to_tensor(Ys, dtype=tf.float32)
Ys_tensor = tf.reshape(Ys_tensor,(1,num_of_recurrences))
# Normal MSE just adds up to one value
loss = tf.reduce_mean(tf.square(Ys_tensor - Y_hat))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_hat * tf.log(Ys_tensor)))
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y_hat, logits=Ys_tensor)
#loss = cross_entropy
# Optimizer:
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(tf.global_variables_initializer())
file_writer = tf.summary.FileWriter('./summary_data_for_tensorboard', sess.graph)

# Training
# Generate data:


index = 0
for epoch in range(num_of_epochs):

    # unpack random training data
    X_raw, Y_raw, index = generate_data_sin(num_of_recurrences, index, echo_step)
    #print("x_raw")
    #print(X_raw)
    #print("y_raw")
    #print(Y_raw)
    X_raw = np.reshape(X_raw, (size_of_input_x, num_of_recurrences, 1))
    Y_raw = np.reshape(Y_raw, (size_of_output_y, num_of_recurrences))

    # fit variables into tensorflow:
    [cur_traing, cur_cost, cur_Y] = sess.run([train, loss, Ys_tensor], {X: X_raw, Y_hat: Y_raw})
    print("Epoch number %s, loss is %s" % (epoch, cur_cost))

[cur_cost, cur_Y] = sess.run([loss, Ys_tensor], {X: X_raw, Y_hat: Y_raw})
print("y hat =")
print(Y_raw)
print("y =")
print(cur_Y)


print('Evaluation')
for i in range(10):
    print(".")


X_raw, Y_raw, idx = generate_data_sin(num_of_recurrences, 16, echo_step)
X_raw = np.reshape(X_raw, (size_of_input_x, num_of_recurrences, 1))
Y_raw = np.reshape(Y_raw, (size_of_output_y, num_of_recurrences))
[cur_X, cur_Y_hat, cur_Y] = sess.run([X,Y_hat, Ys_tensor], {X: X_raw, Y_hat: Y_raw})


cur_Y_hat = np.reshape(cur_Y_hat, (-1))
cur_Y = np.reshape(cur_Y, (-1))
cur_X = np.reshape(cur_X, (-1))
label_Y_hat, = plt.plot(cur_Y_hat, label='cur_Y_hat')
label_cur_Y, = plt.plot(cur_Y, label='cur_Y')
label_cur_X, = plt.plot(cur_X, label='cur_X')
plt.legend(handles=[label_Y_hat, label_cur_Y, label_cur_X])
plt.draw()
plt.show()








