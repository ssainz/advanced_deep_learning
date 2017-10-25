from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import numpy as np
import pickle
import random

def generate_deep_net_normal_init(size_of_image = 784,
                                  size_of_image_features=1,
                      size_of_classes = 10,
                      size_of_first_convolution_filter = 5,
                      size_of_features_in_first_convolution = 32,
                      size_of_second_convolution_filter = 5,
                      size_of_features_in_second_convolution = 64,
                      cur_padding='VALID',
                      max_pool_stride=[1, 2, 2, 1],
                      size_of_flattened_parameters=4 * 4 * 64,
                      size_of_dense_layer=1024):


    x = tf.placeholder(tf.float32, [None, size_of_image])
    y_hat = tf.placeholder(tf.float32, [None, size_of_classes])

    # The actual image to enter the convolution.
    # First dim is the batch size.
    # Second dim and third dim is the image (length and width).
    # Fourth dim is the features size.
    x_image = tf.reshape(x, [-1, int(math.sqrt(size_of_image/size_of_image_features)), int(math.sqrt(size_of_image/size_of_image_features)), size_of_image_features])

    # Create the weights for the filters.
    # The first two dimensions are the size of the 2-D filter.
    # The third dimension is the number of channels in the image (say 3 if it is RGB).
    # The fourth dimension is the output features (how many features).
    W_conv1 = tf.Variable(tf.random_normal(
        shape=[size_of_first_convolution_filter, size_of_first_convolution_filter, size_of_image_features,
               size_of_features_in_first_convolution], stddev=0.1))
    # The bias equal to the number of output features.
    b_conv1 = tf.Variable(tf.random_normal( shape=[size_of_features_in_first_convolution], stddev=0.1))
    # Convolution. The strides is one number per dimension:
    #    1st dimension is the batch size, second and third dimension is the image, fourth dim is the features
    # The padding is "VALID" which means no padding.
    # Or, "SAME" which is explained here: https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
    # "SAME" padding means that the size of the image after convolution will be the same as the image itself!!
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding=cur_padding) + b_conv1)
    # Max pool. For both the ksize (window size of the max pool, and the strides:
    #  The first dimension is the batch number , 2nd and 3rd dims are the image, 4th dimension is feature number.
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=max_pool_stride, padding='SAME')

    W_conv2 = tf.Variable(tf.random_normal(shape=[size_of_second_convolution_filter, size_of_second_convolution_filter,
                                                   size_of_features_in_first_convolution,
                                                   size_of_features_in_second_convolution], stddev=0.1))
    b_conv2 = tf.Variable(tf.random_normal(shape=[size_of_features_in_second_convolution], stddev=0.1))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding=cur_padding) + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=max_pool_stride, padding='SAME')

    # flatten
    W_flatten = tf.Variable(
        tf.random_normal(shape=[size_of_flattened_parameters, size_of_dense_layer], stddev=0.1))
    b_flatten = tf.Variable(tf.random_normal(shape=[size_of_dense_layer],stddev=0.1))
    flatten_pool = tf.reshape(h_pool2, [-1, size_of_flattened_parameters])

    # apply fully connected layer of 1024 neurons
    fc7 = tf.nn.relu(tf.matmul(flatten_pool, W_flatten) + b_flatten)

    # Output layer:
    W_output = tf.Variable(tf.random_normal(shape=[size_of_dense_layer, size_of_classes], stddev=0.1))
    b_output = tf.Variable(tf.random_normal(shape=[size_of_classes], stddev=0.1))

    # Apply output layer
    y = tf.matmul(fc7, W_output) + b_output

    # softmax to the final 10 classes (mnist)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_hat, logits=y)
    loss = tf.reduce_mean(cross_entropy)

    # Optimizer:
    # learning rate:
    eta = 1e-4
    train = tf.train.AdamOptimizer(eta).minimize(loss)

    # Accuracy:
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (train, loss, accuracy, x, y_hat, y, W_conv1, W_conv2)


# decompress Cifar data.
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Prepares the images from a list of 10,000 flat arrays of size 3072 each, into 10,000 tensors of size 32,32,3
def cifar_from_vector_to_tensor(image_10000_3072):

    number_of_images = len(image_10000_3072[:,0])
    image_10000_3_32_32 = np.reshape(image_10000_3072,(number_of_images,3,32,32))
    image_10000_32_32_3 = image_10000_3_32_32.transpose([0,2,3,1])
    # We revert back to a image that can be trivially reshaped by the convolution.
    image_10000_3072_trivial = np.reshape(image_10000_32_32_3, (number_of_images,3072))
    return image_10000_3072_trivial

# Prepares expected class (y hat) from a list of numbers to a boolean array of 10 X number of samples
def cifar_from_categorical_to_vector(classes_10000):
    number_of_images = len(classes_10000)
    y = np.zeros((number_of_images, 10))
    row_index = range(number_of_images)
    y[row_index,classes_10000] = 1
    return y

# Prints feature maps and filters. It prints a feature map in a row.
def print_convolutions(convolution):
    for convolution_name in convolutions.keys():
        # For each convolution, go over its feature maps:
        convolution = convolutions[convolution_name]
        feature_maps = []
        filter_size = convolution.shape[0]
        number_of_input_features = convolution.shape[2]
        number_of_feature_maps = convolution.shape[3]
        for _feature in range(number_of_feature_maps):
            feature_map = convolution[:,:,:,_feature]
            # Now we have one feature map.
            # Transform from a shape=[filter_size,filter_size,input_features] to: new_shape=[filter_size, filter_size * input_features]
            row_based_feature_map = np.reshape(feature_map.transpose(), (filter_size * number_of_input_features, filter_size)).transpose()
            feature_maps.append(row_based_feature_map)
        feature_maps_image = np.stack(feature_maps)
        # reshape to a 2D tensor (after the stack operation, we end up with an array of shape [number_of_feature_maps, filter_size, filter_size * number_of_input_features]
        # we want a 2D tensor shape number_of_feature_maps * filter_size, filter_size * number_of_input_features
        feature_maps_image = np.reshape(feature_maps_image, (number_of_feature_maps * filter_size, filter_size * number_of_input_features))

        # Printing the feature maps (one feature map per row)
        subp = plt.subplot(1, 1, 1)
        title = "Feature map for %s" % (convolution_name)
        subp.set_title(title)
        subp.imshow(feature_maps_image)
        plt.draw()
        plt.show()

# First prepare the network
train, loss, accuracy, x, y_hat, y, W_conv1, W_conv2= generate_deep_net_normal_init(size_of_image = 3072,
                  size_of_image_features=3,
                  size_of_classes = 10,
                  size_of_first_convolution_filter = 5,
                  size_of_features_in_first_convolution = 100,
                  size_of_second_convolution_filter = 5,
                  size_of_features_in_second_convolution = 100,
                  cur_padding='VALID',
                  max_pool_stride=[1, 2, 2, 1],
                  size_of_flattened_parameters= 5 * 5 * 100,
                  size_of_dense_layer=1024)

# Start tensor flow:
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Load dataset
path_loc = "/srv/datasets/cifar/"
training_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4"]
testing_file = "data_batch_5"




training_loss = []
training_accuracy = []
iteration = 0
for j in range(1):
    for file in training_files:
        images = unpickle(path_loc + file)
        x_raw_tensor = cifar_from_vector_to_tensor(images[b'data'])
        y_raw_vector = cifar_from_categorical_to_vector(images[b'labels'])
        number_of_samples = len(images[b'labels'])
        # Here we provide the training in mini-batch of 50 to 50
        step = 10
        indexx = 0
        for i in range(int(math.ceil(number_of_samples/step))):
            # Process and create minibatch X, Y
            upper_index = indexx + step
            if upper_index > number_of_samples:
                upper_index = number_of_samples
            x_raw_mini = x_raw_tensor[indexx:upper_index]
            y_raw_mini = y_raw_vector[indexx:upper_index]
            # update indexx for next iteration
            indexx = upper_index
            # Training
            [cur_train, cur_loss, cur_accuracy] = sess.run([train, loss, accuracy], feed_dict={x: x_raw_mini, y_hat: y_raw_mini})
            if iteration % 500 == 0:
                # Print every X iterations
                print("Iteration (%s), loss (%s), accuracy (%s)" % (iteration, cur_loss, cur_accuracy))
                training_loss.append(cur_loss)
                training_accuracy.append(cur_accuracy)
            iteration = iteration + 1


# Load test data
images = unpickle(path_loc + testing_file)
x_raw_tensor = cifar_from_vector_to_tensor(images[b'data'])
y_raw_vector = cifar_from_categorical_to_vector(images[b'labels'])
number_of_samples = len(images[b'labels'])

# Load lookup values:
md = unpickle(path_loc + "batches.meta")
label_names = md[b'label_names']

print("Print feature maps and filters")
[testing_y, testing_y_hat, final_W_conv1, final_W_conv2] =  sess.run([y, y_hat, W_conv1, W_conv2], feed_dict={x: x_raw_tensor, y_hat: y_raw_vector})

# print convolutions
convolutions = {"First layer": final_W_conv1, "Second layer": final_W_conv2}
print_convolutions(convolutions)

print("Print 3 samples")
[testing_y, testing_y_hat] =  sess.run([y, y_hat], feed_dict={x: x_raw_tensor, y_hat: y_raw_vector})

fig, plot_handle = plt.subplots(1,1)
fig.suptitle("CIFRA 10 samples (actual vs predicted)")
for i in range(3):
    random_image_index = random.randint(0,len(x_raw_tensor[:,0]))
    random_image = np.reshape(x_raw_tensor[random_image_index], (32,32,3))
    subp = plt.subplot(1,1,1)
    title = "Actual Label: %s, Predicted Label: %s" % (label_names[np.argmax(testing_y_hat[random_image_index])], label_names[np.argmax(testing_y[random_image_index])])
    subp.set_title(title)
    subp.imshow(random_image)
    plt.draw()
    plt.show()


print("Print accuracy in test data")
print(sess.run(accuracy, feed_dict={x: x_raw_tensor, y_hat: y_raw_vector}))

plt.subplot(1, 2, 1)
label_training_cost, = plt.plot(np.array(training_loss), label='training_loss_random_weights')
plt.legend(handles=[label_training_cost])
plt.subplot(1, 2, 2)
label_training_accuracy, = plt.plot(np.array(training_accuracy), label='training_accuracy_random_weights')
plt.legend(handles=[label_training_accuracy])
plt.draw()
plt.show()

