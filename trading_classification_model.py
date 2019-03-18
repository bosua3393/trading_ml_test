import tensorflow as tf
from data import btc_sorted

batch_x, batch_label = btc_sorted.batch_x, btc_sorted.batch_label

n_input = 300
n_hidden1 = 100
n_hidden2 = 100
n_hidden3 = 100
n_output = 2

x = tf.placeholder(tf.float32, [None, n_input], name='x')
label = tf.placeholder(tf.float32, [None, n_output], name='y')

with tf.name_scope(name='layer1'):
    w1 = tf.Variable(tf.random_uniform((n_input, n_hidden1), -1, 1), name='weight1')
    b1 = tf.Variable(tf.random_uniform((1, n_hidden1), -1, 1), name='bias1')
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1, name='output1')

with tf.name_scope(name='layer2'):
    w2 = tf.Variable(tf.random_uniform((n_hidden1, n_hidden2), -1, 1), name='weight2')
    b2 = tf.Variable(tf.random_uniform((1, n_hidden2), -1, 1), name='bias2')
    y2 = tf.nn.relu(tf.matmul(y1, w2) + b2, name='output2')

with tf.name_scope(name='layer3'):
    w3 = tf.Variable(tf.random_uniform((n_hidden2, n_hidden3), -1, 1), name='weight3')
    b3 = tf.Variable(tf.random_uniform((1, n_hidden3), -1, 1), name='bias3')
    y3 = tf.nn.relu(tf.matmul(y2, w3) + b3, name='layer3')

with tf.name_scope(name='layer4'):
    w4 = tf.Variable(tf.random_uniform((n_hidden3, n_output), -1, 1), name='weight4')
    b4 = tf.Variable(tf.random_uniform((1, n_output), -1, 1), name='bias4')
    y4 = tf.nn.softmax(tf.matmul(y3, w4) + b4, name='layer4')

correct_pred = tf.equal(tf.argmax(y4, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./best_model/model.ckpt")
    print(sess.run(accuracy, {x: batch_x, label:batch_label}))
