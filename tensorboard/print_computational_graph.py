import tensorflow as tf
sess = tf.Session()

#file_writer = tf.summary.FileWriter('./summary_data_for_tensorboard', sess.graph)
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

#sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3:", node3)

file_writer = tf.summary.FileWriter('./summary_data_for_tensorboard', sess.graph)
print("sess.run(node3):", sess.run(node3))


# run in tensorboard with:
# tensorboard --logdir=path/to/log-directory