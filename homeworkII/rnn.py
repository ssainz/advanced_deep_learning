import tensorflow as tf
import numpy as np

num_batches = 20
batch_size = 10
num_features = 10
total_num_features = 200

#Preprocess the input: from (total_samples, total_features) to (num_batches, ceil(total_number_features / num_features), batch_size, num_features)
initial_dataset = np.reshape(np.array(range(100)), (5,20))
prepared = np.reshape(initial_dataset, (5,4,-1))



words_in_dataset = tf.placeholder(tf.float32, [num_batches, batch_size, num_features])
