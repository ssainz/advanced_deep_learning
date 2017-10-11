import tensorflow as tf
sess = tf.Session()


# first dimension are the input size of each x_i vector,
# second dimension is the t_i (how many recurrences),
# the third dimension is number of samples.
X = tf.placeholder(tf.float32, shape=(2,3,1))
# first dimension is size of y,
# second dimension is how many recurrencies
# third dimension is how many samples
Y_hat = tf.placeholder(tf.float32, shape=(1,3))


h_t = tf.Variable(tf.random_uniform([3,1], minval=0.0, maxval=0.3), name="h_t")
W_i = tf.Variable(tf.random_uniform([3, 2], minval=0.0, maxval=0.3), name="W_i")
W_h = tf.Variable(tf.random_uniform([3,3], minval=0.0, maxval=0.3), name="W_h")
W_o = tf.Variable(tf.random_uniform([1,3], minval=0.0, maxval=0.3), name="W_o")

X_is = tf.unstack(X, axis=1)
Ys = []
for x_i in X_is:
    h_t = tf.sigmoid(tf.matmul(W_i, x_i) + tf.matmul(W_h, h_t))
    y_i = tf.nn.softmax(tf.matmul(W_o,h_t))
    Ys.append(y_i)

Ys_tensor = tf.convert_to_tensor(Ys, dtype=tf.float32)
Ys_tensor = tf.reshape(Ys_tensor,(1,3))
Loss = tf.reduce_mean(tf.square(Ys_tensor - Y_hat))

sess.run(tf.global_variables_initializer())
file_writer = tf.summary.FileWriter('./summary_data_for_tensorboard', sess.graph)




