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
words_placeholder = tf.placeholder(tf.float32, [batch_size, time_steps])

# Placeholder for the classes classification
y_hat = tf.placeholder(tf.int32, [batch_size, num_classes])

# LSTM cell (this helps you when building the RNN)
lstm = tf.contrib.rnn.LSTMCell(hidden_unit_size)

# Initialize the random weights.
#h_0 =  lstm.zero_state(batch_size, tf.float32)
h_0 = tf.zeros([batch_size, hidden_unit_size], dtype=tf.float32)
h_t = tf.zeros([batch_size, hidden_unit_size], dtype=tf.float32)
state = h_t, h_0
# h_t = h_0
for i in range(time_steps):
    # Link all the cells together (each iteration is a new time)
    input_tensor = tf.expand_dims(words_placeholder[:, i], -1) # Now the input_sensor is shape (batch_size, 1)
    h_output, state = lstm(input_tensor, state)

    # Here we can also store the outputs into some array and later on sum them up, outside of the loop,
    # to calculate the loss.

# In our case, we just care about the final step.
h_final = state
h_final_output = h_output

# Then initialize softmax
W_softmax = tf.Variable(tf.random_normal(shape=[hidden_unit_size, num_classes], stddev=0.1))
b_softmax = tf.Variable(tf.random_normal(shape=[num_classes], stddev=0.1))

# Multiply to reduce to number of classes. h_output is [batch_size, hidden_unit_size], W_softmax is [hidden_unit_size, num_classes]
classes_before_softmax = tf.matmul(h_final_output, W_softmax) + b_softmax

# Loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_hat, logits=classes_before_softmax)
loss = tf.reduce_mean(cross_entropy)
# Optimizer:
# learning rate:
eta = 1e-4
train = tf.train.AdamOptimizer(eta).minimize(loss)


# Accuracy:
y = tf.nn.softmax(classes_before_softmax)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Start states
cur_state_h_0 = h_0.eval(session=sess)
cur_state_h_t = h_t.eval(session=sess)
cur_state = cur_state_h_t, cur_state_h_0
i = 0

for current_batch_of_words in words_batches:
    one_hot_y_yat = np.zeros([batch_size,num_classes])
    for j in range(batch_size):
        one_hot_y_yat[j, np.random.randint(num_classes)] = 1
    print(one_hot_y_yat)
    cur_state, current_loss, cur_train = sess.run([h_final, loss, train],
        # Initialize the LSTM state from the previous iteration.
        feed_dict={state: cur_state, words_placeholder: current_batch_of_words, y_hat: one_hot_y_yat})
    print("It: (%s) : loss = %s" % (i, current_loss))
    i = i + 1
