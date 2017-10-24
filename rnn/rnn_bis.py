import tensorflow as tf
import numpy as np

num_batches = 33
batch_size = 4
time_steps = 5
hidden_unit_size = 20
num_classes = 2

words = np.random.randint(0, 9, size=num_batches * batch_size * time_steps)

words_batches = np.reshape(words, (num_batches, batch_size, time_steps))

# Placeholder for the inputs in a given iteration.
words_placeholder = tf.placeholder(tf.int32, [batch_size, time_steps])

# Placeholder for the classes classification
y_hat = tf.placeholder(tf.int32, [batch_size, num_classes])

# LSTM cell (this helps you when building the RNN)
lstm = tf.contrib.rnn.BasicLSTMCell(hidden_unit_size)

# Initialize the random weights.
h_0 =  lstm.zero_state(batch_size, tf.float32)
h_t =  tf.Variable(tf.zeros([batch_size, hidden_unit_size]))
for i in range(time_steps):
    # Link all the cells together (each iteration is a new time)
    h_output, h_t = lstm(words_placeholder[:, i], h_t)

    # Here we can also store the outputs into some array and later on sum them up, outside of the loop,
    # to calculate the loss.

# In our case, we just care about the final step.
h_final = h_t
h_final_output = h_output

# Then initialize softmax
W_softmax = tf.Variable(tf.random_normal(batch=[hidden_unit_size, num_classes], stddev=0.1))
b_softmax = tf.Variable(tf.random_normal(batch=[num_classes], stddev=0.1))

# Multiply to reduce to number of classes. h_output is [batch_size, hidden_unit_size], W_softmax is [hidden_unit_size, num_classes]
classes_before_softmax = tf.matmul(h_final_output, W_softmax) + b_softmax

# Loss
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_hat, logits=classes_before_softmax)

# Optimizer:
# learning rate:
eta = 1e-4
train = tf.train.AdamOptimizer(eta).minimize(loss)


# Accuracy:
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# A numpy array holding the state of LSTM after each batch of words.
cur_state = h_0.eval()
total_loss = 0.0
i = 0

for current_batch_of_words in words_batches:
    dummy_y_hat = np.random.randint(0, 3, size=num_classes)
    cur_state, current_loss, cur_train = session.run([h_final, loss, train],
        # Initialize the LSTM state from the previous iteration.
        feed_dict={h_0: cur_state, words_placeholder: current_batch_of_words, y_hat: dummy_y_hat})
    total_loss += current_loss
    print("It: (%s) : loss = %s" % (i, current_loss))
    i = i + 1
