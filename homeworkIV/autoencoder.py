import tensorflow as tf
import numpy as np
import random
import heapq
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA


def get_training_data():
    mnist = input_data.read_data_sets("MNIST_data/")

    # 1.1 Get the first 1000 images for every number.
    X_raw = mnist.train.images
    Y_raw = mnist.train.labels

    first_1000_elements_in_mnist = []
    for i in range(11):
        indx = np.nonzero(Y_raw == i)[0]
        #print(len(indx))
        first_1000_elements_in_mnist.append(indx[0:1000])

    dataset_idx = np.concatenate(first_1000_elements_in_mnist)
    # print(len(dataset_idx))
    # print(np.shape(X_raw[0]))

    #print("DATASET_", dataset_idx)
    #print("DATASETs_", dataset_idx.size)

    # 2. Create random training sets.
    np.random.shuffle(dataset_idx)

    return (dataset_idx, X_raw, Y_raw)

def stochastic_gradient_backprogation(sess, number_of_iterations, x, x_hat, loss, train, dataset_idx, X_raw, Y_raw, batch_size):

    i = 0
    it = 0
    losses = []
    for j in range(number_of_iterations):
        i = 0
        while i < dataset_idx.size:
            X_batch = X_raw[dataset_idx[i:batch_size+i]]
            #print(X_batch)
            cur_loss, cur_train = sess.run([loss, train], {x: X_batch})
            #print(it," ", (it+1) % 20)
            if (it+1) % 100 == 0:
                losses.append(cur_loss)
                #print("loss, ", cur_loss)
            it += 1
            i = i + batch_size

    return losses

def get_latent_space_autoencoder(sess, x, dataset_idx, X_raw, h_x):

    X_batch = X_raw[dataset_idx[:]]
    [cur_h_x] = sess.run([h_x], {x: X_batch})

    #print("X_BATCH ", X_batch.shape)
    #print("H_x ", cur_h_x.shape)

    return cur_h_x

def get_latent_space_pca(dataset_idx, X_raw, num_components):
    new_ds = X_raw[dataset_idx]

    pca = PCA(n_components=num_components)
    pca.fit(new_ds)

    return pca.transform(new_ds)

#def get_precision_recall_of_items_closes_to_K(K=random_item, X_raw=X_raw, Y_raw=Y_raw, closes_n_items=50, latent_space=latent_spaces)
def get_precision_recall_of_items_closest_to_K(K, dataset_idx, X_raw, Y_raw, closes_n_items, latent_spaces, neighborhood_size):

    #A. Get the K item.
    # print("PRC RECL")
    # print(len(latent_spaces))
    # print(latent_spaces.shape)
    # print(K)
    K_item = latent_spaces[K]
    classes = Y_raw[dataset_idx[:]]
    K_class = classes[K]

    #B. Iterate over the list and keep the smallest 50.
    heap = []
    i = 0
    for space in latent_spaces:
        if len(heap) > neighborhood_size+1:
            heap.pop()
        L2_distance = np.sum((K_item - space)**2)
        heapq.heappush(heap, (L2_distance,classes[i]))
        i += 1
    heapq.heappop(heap) # remove the smallest one (which is itself)

    #C. Iterate over the 50 elements and calculate precision and recall:
    # precision= tp / tp+fp
    # recall= tp / tp+tn


    actual_class = []
    while len(heap) > 0:
        (cur_L2, cur_class) = heapq.heappop(heap)
        actual_class.append(cur_class)

    actual_class = np.array(actual_class)
    predicted_class = np.ones(len(actual_class), dtype=np.int32) * K_class

    # print(actual_class)
    # print(predicted_class)
    # print(np.nonzero(actual_class == predicted_class)[0])
    tp = len(np.nonzero(actual_class == predicted_class)[0])
    tn = 0 # because we classify as all the same as the K class
    fn = 0 # because we classify all as the same class as K class
    fp = len(np.nonzero(actual_class != predicted_class)[0])

    # print(tp)
    # print(fp)

    return (tp/(tp+fp)), (tp/tp+fn)





# 3. Create network
def build_encoder_network(image_size, number_hidden_units):
    x = tf.placeholder(tf.float32, [None, image_size])

    W = tf.Variable(tf.random_normal(shape=[image_size, number_hidden_units], stddev=0.1))

    b = tf.Variable(tf.random_normal(shape=[1, number_hidden_units], stddev=0.1))

    h_x = tf.nn.relu(tf.matmul(x, W) + b) # h_x is shape [None, number_hidden_units]

    W_t = tf.transpose(W)

    c = tf.Variable(tf.random_normal(shape=[1, image_size], stddev=0.1))

    x_hat = tf.nn.sigmoid(tf.matmul(h_x, W_t) + c) # x_hat is shape [None, image_size]

    loss = tf.reduce_sum(tf.squared_difference(x, x_hat), axis=1) / 2 # As defined by Lecture 8 slide 5 for real-valued inputs
    # At thsi point loss is shape [None,1] or [None].

    loss = tf.reduce_mean(loss)

    # Adam optimizer:
    eta = 0.0001
    opt = tf.train.AdamOptimizer(learning_rate=eta)
    train = opt.minimize(loss)


    return (x, x_hat, loss, train, h_x)



# 1. Import the MNIST
dataset_idx, X_raw, Y_raw = get_training_data()

#print("DATASET_", dataset_idx)
#print("DATASETs_", dataset_idx.size)

# 4. Loop through the hyperparameters
hyperparameter_N = [2,5,10,20,50,100]
networks = []
for N in hyperparameter_N:
    networks.append((build_encoder_network(image_size=28*28, number_hidden_units=N)))

# Init TF
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


# In case we want to restore previous model, just use:
# 1) iterations set to 0
# 2) restore:
# saver.restore(sess, "/srv/datasets/autoencoder/model.ckpt")
# print("model restored")


# Save model for later :)
saver = tf.train.Saver()


# 4.2 train the networks:
losses_list = []
batch_size = 10
num_iter = 50
for network in networks:
    (cur_x, cur_x_hat, cur_loss, cur_train, cur_h_x) = network
    losses = stochastic_gradient_backprogation(sess, number_of_iterations=num_iter, x=cur_x, x_hat=cur_x_hat, loss=cur_loss, train=cur_train, dataset_idx=dataset_idx, X_raw=X_raw, Y_raw=Y_raw, batch_size=batch_size)
    losses_list.append(losses)
    print(" losses: ", losses)


#4.3 Now we get all the image's latent space (h_x) and compare with PCA::
random_item = random.randint(0, len(dataset_idx))
latent_spaces_list = []
batch_size = 10
i = 0
neighborhood_size = 50
for network in networks:

    # First autoencoder.
    (cur_x, cur_x_hat, cur_loss, cur_train, cur_h_x) = network
    latent_spaces = get_latent_space_autoencoder(sess, cur_x, dataset_idx, X_raw, cur_h_x)
    latent_spaces_list.append(latent_spaces)
    precision, recall = get_precision_recall_of_items_closest_to_K(K=random_item, dataset_idx=dataset_idx, X_raw=X_raw, Y_raw=Y_raw, closes_n_items=50, latent_spaces=latent_spaces, neighborhood_size=neighborhood_size)
    print("Autoencoder: N=%s, precision=%s, recall=%s" %(hyperparameter_N[i], precision, recall))

    # Second, PCA:
    latent_space_pca = get_latent_space_pca(dataset_idx, X_raw, hyperparameter_N[i])
    precision, recall = get_precision_recall_of_items_closest_to_K(K=random_item, dataset_idx=dataset_idx, X_raw=X_raw,
                                                                   Y_raw=Y_raw, closes_n_items=50,
                                                                   latent_spaces=latent_space_pca,
                                                                   neighborhood_size=neighborhood_size)
    print("PCA        : N=%s, precision=%s, recall=%s" % (hyperparameter_N[i], precision, recall))


    i += 1


# 5. Print results.
save_path = saver.save(sess, "/srv/datasets/autoencoder/model.ckpt")
print("Model saved in file: %s" % save_path)