"""
@Created on: 17/05/2017,
@author: Umesh Kumar,
@version: 0.0.1


Sphinx Documentation:

"""
import tensorflow as tf

from rztdl.utils.file import read_csv

data_path = "../data/iris_data_multiclass.csv"
train_data, train_label, valid_data, valid_label, test_data, test_label = read_csv(data_path, split_ratio=[70, 20, 10],
                                                                                   delimiter=",",
                                                                                   normalize=False,
                                                                                   randomize=True)
learning_rate = 0.01
epoch = 150
display_step = 1
beta = 0.01  # Regularisation parameter

x = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

weights = {
    'wout': tf.Variable(tf.random_uniform([4, 1], minval=-1, maxval=1), dtype=tf.float32)
}

bias = {
    'bout': tf.Variable(tf.zeros([1]), dtype=tf.float32)
}


def model(x, weights, bias):
    return tf.sigmoid(tf.add(tf.matmul(x, weights['wout']), bias['bout']))


pred = model(x, weights, bias)

# normal cost function
cost = tf.reduce_mean(tf.square(y - pred))

# Loss function using L2 Regularization
regularizer = tf.nn.l2_loss(weights['wout'])
loss = tf.reduce_mean(cost + beta * regularizer)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# accuracy
correct_pred = tf.equal(tf.greater(pred, 0.5), tf.greater(y, 0.5))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        _, c, p, acc = sess.run([optimizer, loss, pred, accuracy], feed_dict={x: train_data, y: train_label})
        if i % display_step == 0:
            print('Epoch: ', i, ' Cost:', c, 'Accuracy:', acc)
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
