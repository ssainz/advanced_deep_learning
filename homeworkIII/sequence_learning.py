import tensorflow as tf
import numpy as np
import math



class LSTM_CELL:

    def __init__(self, size_of_input_x, size_of_output_y, size_of_h_t, time_steps):

        # RNN weights
        # Input gate:
        self.W_x_i = tf.Variable(tf.random_normal([size_of_h_t, size_of_input_x], stddev=0.1), name="W_x_i")
        self.W_h_i = tf.Variable(tf.random_normal([size_of_h_t, size_of_h_t], stddev=0.1), name="W_h_i")
        self.W_prev_h_i = tf.Variable(tf.random_normal([size_of_h_t, size_of_h_t], stddev=0.1), name="W_prev_h_i")
        self.b_i = tf.Variable(tf.random_normal([size_of_h_t, 1], stddev=0.1), name="b_i")

        # New state gate:
        self.W_x_c = tf.Variable(tf.random_normal([size_of_h_t, size_of_input_x], stddev=0.1), name="W_x_c")
        self.W_h_c = tf.Variable(tf.random_normal([size_of_h_t, size_of_h_t], stddev=0.1), name="W_h_c")
        self.W_prev_h_c = tf.Variable(tf.random_normal([size_of_h_t, size_of_h_t], stddev=0.1), name="W_prev_h_c")
        self.b_c = tf.Variable(tf.random_normal([size_of_h_t, 1], stddev=0.1), name="b_c")

        # Forget gate:
        self.W_x_f = tf.Variable(tf.random_normal([size_of_h_t, size_of_input_x], stddev=0.1), name="W_x_f")
        self.W_h_f = tf.Variable(tf.random_normal([size_of_h_t, size_of_h_t], stddev=0.1), name="W_h_f")
        self.W_prev_h_f = tf.Variable(tf.random_normal([size_of_h_t, size_of_h_t], stddev=0.1), name="W_prev_h_f")
        self.b_f = tf.Variable(tf.random_normal([size_of_h_t, 1], stddev=0.1), name="b_f")

        # LSTM cell output gate:
        self.W_x_o = tf.Variable(tf.random_normal([size_of_h_t, size_of_input_x], stddev=0.1), name="W_x_o")
        self.W_h_o = tf.Variable(tf.random_normal([size_of_h_t, size_of_h_t], stddev=0.1), name="W_h_o")
        self.W_prev_h_o = tf.Variable(tf.random_normal([size_of_h_t, size_of_h_t], stddev=0.1), name="W_prev_h_o")
        self.b_o = tf.Variable(tf.random_normal([size_of_h_t, 1], stddev=0.1), name="b_o")

        # Actual output weights
        self.W_out = tf.Variable(tf.random_normal([size_of_output_y, size_of_h_t], stddev=0.1), name="W_out")
        self.b_out = tf.Variable(tf.random_normal([size_of_output_y, 1], stddev=0.1), name="b_out")


    def create_lstm_cell_hidden_layer(self, h_t, h_prev_layer, x_t, C_t):
        forget = tf.sigmoid(tf.matmul(self.W_x_f, x_t) + tf.matmul(self.W_h_f, h_t) + tf.matmul(self.W_prev_h_f, h_prev_layer) + self.b_f)
        input = tf.sigmoid(tf.matmul(self.W_x_i, x_t) + tf.matmul(self.W_h_i, h_t) + tf.matmul(self.W_prev_h_i, h_prev_layer) + self.b_i)
        new_state = tf.sigmoid(tf.matmul(self.W_x_c, x_t) + tf.matmul(self.W_h_c, h_t) + tf.matmul(self.W_prev_h_c, h_prev_layer) + self.b_c)
        output = tf.sigmoid(tf.matmul(self.W_x_o, x_t) + tf.matmul(self.W_h_o, h_t) + tf.matmul(self.W_prev_h_o, h_prev_layer) + self.b_o)
        C_new = tf.tanh(new_state)
        self.C_t = tf.multiply(forget, C_t) + tf.multiply(input, C_new)
        self.h_t = tf.multiply(output, tf.tanh(self.C_t))

        return (self.h_t, self.C_t)

    def create_lstm_cell_first_layer(self, h_t, x_t, C_t):
        forget = tf.sigmoid(tf.matmul(self.W_x_f, x_t) + tf.matmul(self.W_h_f, h_t) +  self.b_f)
        input = tf.sigmoid(tf.matmul(self.W_x_i, x_t) + tf.matmul(self.W_h_i, h_t) +  self.b_i)
        new_state = tf.sigmoid(tf.matmul(self.W_x_c, x_t) + tf.matmul(self.W_h_c, h_t) +  self.b_c)
        output = tf.sigmoid(tf.matmul(self.W_x_o, x_t) + tf.matmul(self.W_h_o, h_t) +  self.b_o)
        C_new = tf.tanh(new_state)
        self.C_t = tf.multiply(forget, C_t) + tf.multiply(input, C_new)
        self.h_t = tf.multiply(output, tf.tanh(self.C_t))

        return (self.h_t, self.C_t)


def generate_seq_to_seq_network(size_of_input_x, size_of_output_y, size_of_h_t, time_steps, number_of_stacked_layers, number_of_warming_up_words):

    X = tf.placeholder(shape=(size_of_input_x, time_steps, None), dtype=tf.float32)
    Y_hat = tf.placeholder(shape=(size_of_output_y, time_steps, None), dtype=tf.float32)

    C_0 = tf.placeholder(shape=(number_of_stacked_layers, size_of_h_t, None), dtype=tf.float32)
    h_0 = tf.placeholder(shape=(number_of_stacked_layers, size_of_h_t, None), dtype=tf.float32)

    b_y = tf.Variable(tf.random_normal([size_of_output_y, 1], stddev=0.1), name="b_o")

    C_0s = np.unstack(C_0, axis=0)
    h_0s = np.unstack(h_0, axis=0)

    # Create LSTM cells
    lstm_cells = []
    for i in range(number_of_stacked_layers):
        lstm_cells.append(LSTM_CELL(size_of_input_x, size_of_output_y, size_of_h_t, time_steps))

    # unpack X:
    x_slices_per_time_step = tf.unstack(X, axis=1)  # each slice will be of shape (size_of_input_x, batch_size)

    # unpack Y:
    y_slices_per_time_step = tf.unstack(Y_hat, axis=1)  # each slice will be of shape (size_of_input_x, batch_size)

    # init losses array:
    losses = []

    # predicted y
    predicted_ys = []

    # First for each step
    first_time_step = True
    for time_step in range(time_steps):
        x_slice_per_time_step = x_slices_per_time_step[time_step]

        first_stack = True
        # secondly for each stack, starting from first one.
        for stack in range(number_of_stacked_layers):

            # get corresponding cell:
            lstm_cell = lstm_cells[stack]

            # get the h_t and C_t
            if first_time_step:
                # For first time we initialize with the zero'ed init states
                h_t = h_0s[stack]
                C_t = C_0s[stack]
            else:
                h_t = lstm_cell.h_t
                C_t = lstm_cell.C_t

            # apply the linear algebra calculations of LSTM cell
            # If it is first layer, we do not need to get previous layer h_t
            if first_stack:
                h_prev_layer, C_t = lstm_cell.create_lstm_cell_first_layer(h_t, x_slice_per_time_step, C_t)
            else:
                h_prev_layer, C_t = lstm_cell.create_lstm_cell_hidden_layer(h_t, h_prev_layer, x_slice_per_time_step, C_t)

            first_stack = False

        # After all the cells have been processed vertically (that is, within the same time_step)
        # we calculate predicted y for this time step:
        # initialize sum as zeros, dimension is same as the output of one time, that is shape=(size_of_output_y, batch_size)
        sum = tf.zeros_like(y_slices_per_time_step[time_step])
        for stack in range(number_of_stacked_layers):
            # Get the cell:
            lstm_cell = lstm_cells[stack]
            sum = sum + tf.matmul(lstm_cell.W_out, lstm_cell.h_t)
        # Here we must sum b_y to each row
        # sum is (size_of_output_y, batches), b_y is (size_of_output_y, 1)
        sum = tf.add(sum, b_y)

        predicted_y = tf.nn.softmax(sum, dim=0)
        predicted_ys.append(predicted_y)


        probability_actual_given_prediction = tf.log(tf.multiply(predicted_y, y_slices_per_time_step[time_step]))
        losses.append(probability_actual_given_prediction) # losses is a list of tensors, each tensor in this list is shape [size_of_output_y, batches]

        first_time_step = False

    # Here we just sum all samples (from all batches), over all the time steps.
    # Should we first sumover the times and then average each sample ?
    loss = -1 * tf.reduce_sum(losses)

    # Create an optimizer.
    eta = 0.0001
    opt = tf.AdamOptimizer(learning_rate=eta)

    # Compute the gradients for a list of variables.
    grads_and_vars = opt.compute_gradients(loss)

    # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
    # need to the 'gradient' part, for example cap them, etc.
    capped_grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]

    # Ask the optimizer to apply the capped gradients.
    train = opt.apply_gradients(capped_grads_and_vars)

    # Accuracy:
    stacked_predicted_ys = tf.stack(predicted_ys, axis=0)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y_hat, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return X, Y_hat, C_0, h_0, train, predicted_ys, loss, accuracy


def generate_words_one_hot_table(sorted_words_list):

    words_one_hot_table = {}
    size = len(sorted_words_list)
    i = 0
    for word in sorted_words_list:
        one_hot = np.zeros(size, dtype=float)
        one_hot[i] = 1.0
        words_one_hot_table[word] = one_hot
        i += 1

    return words_one_hot_table


def read_words(file_name):
    words = {}
    with open(file_name, 'rt') as f:
        for line in f:
            tokens = line.split(' ')
            for token in tokens:
                if token in words:
                    words[token] += 1
                else:
                    words[token] = 1

    #print(len(words.keys()))
    words_sorted = [k for k in words.keys()]
    words_sorted.sort()
    #print(words_sorted)

    words_one_hot_table = generate_words_one_hot_table(words_sorted)

    return words_one_hot_table


def read_words_sequence(file_name, words_table):
    sequence = []
    with open(file_name, 'rt') as f:
        for line in f:
            tokens = line.split(' ')
            for token in tokens:
                sequence.append(words_table[token])
    return np.stack(sequence)


def generate_random_sequences_batches(sequences, batch_size, num_of_words_before_prediction_N, number_of_words_to_be_predicted):
    sequence_length = sequences.shape[0]

    # we want to start a sequence in such a way that there are enough words ahead to make the prediction
    valid_lenght = sequence_length - (number_of_words_to_be_predicted + num_of_words_before_prediction_N + 1)

    indexes = np.array(range(valid_lenght))

    # randomly shuffle indexes for training
    np.random.shuffle(indexes)

    # split the indexes in batch size and build a list of (valid_lenght/batch_size) tensors.
    # Each tensor with shape: batch_size, num_of_words_before_prediction_N, number_of_words_to_be_predicted, vocabulary size
    list_of_batches = []
    for i in range(math.floor(valid_lenght/batch_size)):

        # get tensor:
        ind = i * batch_size
        new_batch_indexes = indexes[ ind : ind+batch_size ]
        list_of_sequences = []
        for index in new_batch_indexes:
            # We append new sequence to the list_of_sequences:
            random_sequence = sequences[index: index + (number_of_words_to_be_predicted + num_of_words_before_prediction_N + 1) ]
            list_of_sequences.append(random_sequence)
            #print("sequence shape" )
            #print(random_sequence.shape)
        new_batch = np.stack(list_of_sequences)

        list_of_batches.append(new_batch)
    return list_of_batches


# given a sequence of words: [tree, on, the, hill], it produces the vector with one position shift to the left for forecasting:
# [on, the, hill]
# notice, it also returns the original sequence with the last word cut.
# returns: [tree, on, the], [on, the, hill]
# Actually sequence has the shape of (batch_size, num_of_words_before_prediction_N+number_of_words_to_be_predicted + 1, vocabulary size )
# First, we transpose to shape (num_of_words_before_prediction_N+number_of_words_to_be_predicted + 1, vocabulary_size, batch_size)
# Secondly, we left-shift the matrix on the first dimension (num_of_words_before_prediction_N+number_of_words_to_be_predicted + 1,). This is to achieve the prediction of one word in the future.
# Thirdly we remove the last item on the first dimension (
# FROM:
#  [num_of_words_before_prediction_N+number_of_words_to_be_predicted + 1]
# TO:
#  [num_of_words_before_prediction_N+number_of_words_to_be_predicted])
# Finally, we transpose back to shape (vocabulary_size, num_of_words_before_prediction_N+number_of_words_to_be_predicted, batch_size)
def generate_X_and_Y_hat_for_sequence(sequence):
    #change to shape [num_of_words_before_prediction_N+number_of_words_to_be_predicted + 1, vocabulary_size, batch_size]
    transposed_sequence = np.transpose(sequence, (1, 2, 0))

    #shift to the left on the first dimension the [num_of_words_before_prediction_N+number_of_words_to_be_predicted + 1] dim.
    shift_to_the_left = np.roll(transposed_sequence, -1, axis=0)

    # Size of first dimension.
    leng = shift_to_the_left.shape[0]

    # Remove the last position on first dimension [num_of_words_before_prediction_N+number_of_words_to_be_predicted + 1]
    X_ = transposed_sequence[0:leng-1]
    Y_ = shift_to_the_left[0:leng-1]

    # Transpose to shape [vocabulary_size, num_of_words_before_prediction_N+number_of_words_to_be_predicted, batch_size]
    return np.transpose(X_, (1,0,2)), np.transpose(Y_, (1,0,2))


# First reads the words into one-hot vectors:
file_name = "/srv/datasets/sequence_to_sequence/text_file.txt"
words_table = read_words(file_name)

# print('---')
# for key in words_table:
#     print('%s ' % key)
#     #print(words_table[key])
#     print('---')

# Secondly, read the sequence already encoded in one-hot vectors (shape: training sequence total len, vocabulary size)
training_sequence = read_words_sequence(file_name , words_table)


# Prepare to create one iteration mini batches
# it returns a list of tensors, each tensor has shape (batch_size, num_of_words_before_prediction_N+number_of_words_to_be_predicted, vocabulary size )
batch_size = 10
num_of_words_before_prediction_N = 5
number_of_words_to_be_predicted = 50
list_of_batches = generate_random_sequences_batches(training_sequence, batch_size, num_of_words_before_prediction_N, number_of_words_to_be_predicted)

# prints
#  for batch in list_of_batches:
#     print(batch.shape)

# Build the network
size_of_input_x = 768
size_of_output_y = 768
size_of_h_t = 512
time_steps = num_of_words_before_prediction_N + number_of_words_to_be_predicted
number_of_stacked_layers = 3
(X, Y_hat, C_0, h_0, train, predicted_ys, loss, accuracy) = generate_seq_to_seq_network(size_of_input_x,
                                                                                        size_of_output_y,
                                                                                        size_of_h_t,
                                                                                        time_steps,
                                                                                        number_of_stacked_layers)

# Start training:
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


iterations = 1
# One iteration is considered seeing all the potential sequences at least once in random order.
for it in range(iterations):
    # Generate minibatches of sequences:
    list_of_batches = generate_random_sequences_batches(training_sequence, batch_size, num_of_words_before_prediction_N,
                                                        number_of_words_to_be_predicted)

    # For each minibatch
    for minibatch in list_of_batches:

        # Generate X and Y:
        X_, Y_hat_ = generate_X_and_Y_hat_for_sequence(minibatch)
        # X_ has shape [size_of_input_x, time_steps, batch_size]
        # Y_ has shape [size_of_input_x, time_steps, batch_size]

        # Generate zero state tensors for LSTM cells in the different stacks of LSTM cells:
        C_0_ = np.zeros(shape=(number_of_stacked_layers, size_of_h_t, batch_size), dtype=np.float32)
        h_0_ = np.zeros(shape=(number_of_stacked_layers, size_of_h_t, batch_size), dtype=np.float32)

        # Train:
        (cur_train, cur_loss, cur_accuracy, cur_predicted_ys) = sess.run([train, loss, accuracy, predicted_ys],
                                                                                  {X: X_, Y_hat: Y_hat_, h_0: h_0_, C_0: C_0_})

    if (it + 1) % 5000 == 0:

        print("it (%s), loss(%s), accuracy (%s)" % (it, cur_loss, cur_accuracy) )


print("it (%s), loss(%s), accuracy (%s)" % (it, cur_loss, cur_accuracy) )


# Evaluation



