from __future__ import print_function
import tensorflow as tf
import input_data

mnist = input_data.read_data_sets('MY_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    with tf.name_scope('weights'):
        initial = tf.truncated_normal(shape, stddev=0.1, name="w")
    return tf.Variable(initial)


def bias_variable(shape):
    with tf.name_scope('biase'):
        initial = tf.constant(0.1, shape=shape, name="b")
    return tf.Variable(initial)


def conv2d(x, W):
    with tf.name_scope('conv2d'):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    with tf.name_scope('max_pool'):
    # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 12544],name='x_input') # 56x224
    ys = tf.placeholder(tf.float32, [None, 5],name='y_input')
    keep_prob = tf.placeholder(tf.float32,name='prop')
    x_image = tf.reshape(xs, [-1, 56, 224, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 56x224x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 28x112x32

## conv2 layer ##
W_conv2 = weight_variable([5,5,8,64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 28x112x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 14x56x64

## fc1 layer ##
W_fc1 = weight_variable([14*56*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 14, 56, 64] ->> [n_samples, 14*56*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 14*56*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 5])
b_fc2 = bias_variable([5])
with tf.name_scope('softmax'):
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))       # loss
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
writer = tf.summary.FileWriter("F:\logs", sess.graph)
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 5 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
