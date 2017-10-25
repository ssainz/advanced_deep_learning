import tensorflow as tf
import csv
import numpy as np

def get_one_hot_class(sentiment):
    if sentiment == 'positive':
        return np.array([0,1])
    else:
        return np.array([1,0])


# Load sentences
def load_sentences(sentence_file, words):
    _Y = []
    _X = []
    with open(sentence_file, 'rt') as f:
        for line in f:
            tokens = line.split(',')
            _Y.append(get_one_hot_class(tokens[0]))
            sentence = ','.join(tokens[1:])
            tokens = sentence.split(' ')
            sequence = [words[token] for token in tokens]
            # Here we need to extend if len(sequence) < 100
            _X.append(sequence)


# Load word vectors
def load_word_vector(word_vector_file):
    word_vectors = {}
    with open(word_vector_file, 'rt') as f:
        word_vectors_reader = csv.reader(f, delimiter=',')
        for row in word_vectors_reader:
            # read the word
            word = row[0]
            # read the number vector, although is string initially
            vector_str = row[1:]
            # convert the string vector to fload vector
            vector_float = np.array([float(val) for val in vector_str])
            word_vectors[word] = vector_float
    return word_vectors

# Create Vanilla RNN:
def create_vanilla_rnn(size_of_h_t, size_of_input_x, size_of_output_y, time_steps):

    X = tf.placeholder(shape=(size_of_input_x, time_steps, None))
    Y_hat = tf.placeholder(shape=(size_of_output_y, None))

    # RNN weights
    h_t = tf.Variable(tf.random_normal([size_of_h_t, None], stddev=0.1), name="h_t")
    W_h = tf.Variable(tf.random_normal([size_of_h_t, size_of_h_t], stddev=0.1), name="W_h")
    b_h = tf.Variable(tf.random_normal([size_of_h_t], stddev=0.1), name="b_h")
    W_i = tf.Variable(tf.random_normal([size_of_h_t, size_of_input_x], stddev=0.1), name="W_i")
    b_i = tf.Variable(tf.random_normal([size_of_h_t], stddev=0.1), name="b_i")

    W_o = tf.Variable(tf.random_normal([size_of_output_y, size_of_h_t], stddev=0.1), name="W_o")
    b_o = tf.Variable(tf.random_normal([size_of_output_y], stddev=0.1), name="b_o")

    x_slices_per_time_step = tf.unstack(X, axis=1) # each slice will be of shape (size_of_input_x, None)

    # Create recurrences:
    for x_slice_per_time_step in x_slices_per_time_step:

        # h_t calculates again:
        h_t = tf.tanh(  (tf.matmul(W_h, h_t) + b_h ) + (tf.matmul(W_i, x_slice_per_time_step) + b_i)  )

    # Final prediction:
    y = tf.softmax(tf.matmul(W_o, h_t) + b_o)

    # Calculate loss:
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y_hat, logits=y)
    loss = tf.reduce_mean(cross_entropy)

    # Optimization
    eta = 1e-4
    train = tf.train.AdamOptimizer(eta).minimize(loss)

    # Accuracy:
    correct_prediction = tf.equal(tf.argmax(y, 0), tf.argmax(Y_hat, 0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return (X, Y_hat, train, accuracy, loss)

(X, Y_hat, train, accuracy, loss) = create_vanilla_rnn(size_of_h_t = 20,
                                                       size_of_input_x = 50,
                                                       size_of_output_y = 2,
                                                       time_steps = 100)


# Load word vector file:
words = load_word_vector("/srv/datasets/sentiment-data/word-vectors.txt")

# Load the training set


# Load the testing set