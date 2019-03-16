from data import sorted_data
import tensorflow as tf
from numpy import array_split

data_x, data_label = sorted_data.batch_x, sorted_data.batch_label

section = 10

train_loop = 20000
learn_rate = .001
check_rate = 100

n_input = 300
n_hidden1 = 10
n_hidden2 = 10
n_hidden3 = 10
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

loss = tf.losses.sigmoid_cross_entropy(label, y4)
train_method = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

loss_scalar = tf.summary.scalar("loss", loss)

correct_pred = tf.equal(tf.argmax(y4, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graph', graph=tf.get_default_graph())
    batch_x = array_split(data_x, section)
    batch_label = array_split(data_label, section)
    for step in range(train_loop):
        for batch in range(len(batch_x)):
            sess.run(train_method, {x: batch_x[batch], label: batch_label[batch]})
        if step % check_rate == 0:
            summary = sess.run(loss_scalar, {x: data_x, label: data_label})
            writer.add_summary(summary, step)
            print("Step:", step, " Loss:", sess.run(loss, {x: data_x, label: data_label}), " Accuracy:", sess.run(accuracy, {x: data_x, label: data_label}))
