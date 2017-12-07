import tensorflow as tf
import numpy as np
import heapq
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

def read_words(file_name):
    words = {}
    with open(file_name, 'rt') as f:
        for line in f:
            tokens = line.split(' ')
            embedding = []
            first_token = True
            for token in tokens:
                if first_token:
                    word = token
                    first_token = False
                else:
                    embedding.append(remove_slash_n(token))
            words[word] = np.array(embedding).astype(np.float32)
    return words

def remove_slash_n(stri):
    return stri.replace("\n", "")

def cosine_similarity(a, b):
    a_b = np.sum(np.multiply(a, b))
    a_a = np.sum(np.multiply(a, a))
    b_b = np.sum(np.multiply(b, b))
    return a_b / (np.sqrt(a_a) * np.sqrt(b_b))

def find_x_closest_words(dictionary, word, x):
    res = []
    heap = []
    for w in dictionary.keys():
        if len(heap) > (x + 1):
            heap.pop()
        # Times minus one because heappush keeps the smallests values
        cos_sim = cosine_similarity(dictionary[w], dictionary[word]) * -1
        heapq.heappush(heap, (cos_sim, w))

    for i in heap:
        (cos_sim, w) = i
        res.append(w)
    return res

words_dic = read_words("/srv/datasets/word2vec/vectors_new.txt")

# print(words_dic)
words_samples = ["life", "market", "stanford", "trump", "public"]

neighborhood_size = 20
for w in words_samples:
    neighborhood = find_x_closest_words(dictionary=words_dic, word=w, x=neighborhood_size)
    print("closest words to %s are " % w, " ", neighborhood)


print("life", words_dic["life"])
print("life", words_dic["notoc"])
print("life", words_dic["logos"])

#T-SNE visualization

#1. stack the word embeddings:
X = []
Y = []
for w in words_dic.keys():
    X.append(words_dic[w])
    Y.append(w)
X = np.stack(X)
Y = np.stack(Y)

#print(X)
#print(X.shape)

print("Before TSNE")
tsne = TSNE(n_components=2)
print("During TSNE")
X_embedded = tsne.fit_transform(X)
print("After TSNE")
plt.scatter(X_embedded[:,0],X_embedded[:,1])
print("Before show")

plt.show()
print("After show")