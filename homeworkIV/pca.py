from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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

# 1. Import the MNIST
dataset_idx, X_raw, Y_raw = get_training_data()

new_ds = X_raw[dataset_idx]

print(np.shape(new_ds))

pca = PCA(n_components=2)
pca.fit(new_ds)

print(np.shape(pca.transform(new_ds)))