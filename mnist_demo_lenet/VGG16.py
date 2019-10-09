import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pylab as plt


def gen_conv2d(input, weight, padding="SAME"):
    return tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding=padding)


def gen_pool2d(input, padding="SAME"):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)


def gen_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def gen_biase(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


x = tf.placeholder(tf.float32, shape=[None, 28*28*1], name="input")
x_ = tf.reshape(x, shape=[-1, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, 10])

conv1_1_w = gen_weight([3, 3, 1, 8])
conv1_1_b = gen_biase([8])
conv1_1 = tf.nn.relu(gen_conv2d(x_, conv1_1_w) + conv1_1_b)

conv1_2_w = gen_weight([3, 3, 8, 8])
conv1_2_b = gen_biase([8])
conv1_2 = tf.nn.relu(gen_conv2d(conv1_1, conv1_2_w) + conv1_2_b)

pool1 = gen_pool2d(conv1_2)
#14*14@8

conv2_1_w = gen_weight([3, 3, 8, 16])
conv2_1_b = gen_biase([16])
conv2_1 = tf.nn.relu(gen_conv2d(pool1, conv2_1_w) + conv2_1_b)

conv2_2_w = gen_weight([3, 3, 16, 16])
conv2_2_b = gen_biase([16])
conv2_2 = tf.nn.relu(gen_conv2d(conv2_1, conv2_2_w) + conv2_2_b)

pool2 = gen_pool2d(conv2_2)
#7*7@16

"""
conv3_1_w = gen_weight([3, 3, 128, 256])
conv3_1_b = gen_biase([256])
conv3_1 = tf.nn.relu(gen_conv2d(pool2, conv3_1_w) + conv3_1_b)

conv3_2_w = gen_weight([3, 3, 256, 256])
conv3_2_b = gen_biase([256])
conv3_2 = tf.nn.relu(gen_conv2d(conv3_1, conv3_2_w) + conv3_2_b)

conv3_3_w = gen_weight([3, 3, 256, 256])
conv3_3_b = gen_biase([256])
conv3_3 =  tf.nn.relu(gen_conv2d(conv3_2, conv3_3_w) + conv3_3_b)

pool3 = gen_pool2d(conv3_3)

conv4_1_w = gen_weight([3, 3, 256, 512])
conv4_1_b = gen_biase([512])
conv4_1 = tf.nn.relu(gen_conv2d(pool3, conv4_1_w) + conv4_1_b)

conv4_2_w = gen_weight([3, 3, 512, 512])
conv4_2_b = gen_biase([512])
conv4_2 = tf.nn.relu(gen_conv2d(conv4_1, conv4_2_w) + conv4_2_b)

conv4_3_w = gen_weight([3, 3, 512, 512])
conv4_3_b = gen_biase([512])
conv4_3 = tf.nn.relu(gen_conv2d(conv4_2, conv4_3_w) + conv4_3_b)

pool4 = gen_pool2d(conv4_3)

conv5_1_w = gen_weight([3, 3, 512, 512])
conv5_1_b = gen_biase([512])
conv5_1 = tf.nn.relu(gen_conv2d(pool4, conv5_1_w) + conv5_1_b)

conv5_2_w = gen_weight([3, 3, 512, 512])
conv5_2_b = gen_biase([512])
conv5_2 = tf.nn.relu(gen_conv2d(conv5_1, conv5_2_w) + conv5_2_b)

conv5_3_w = gen_weight([3, 3, 512, 512])
conv5_3_b = gen_biase([512])
conv5_3 = tf.nn.relu(gen_conv2d(conv5_2, conv5_3_w) + conv5_3_b)

pool5 = gen_pool2d(conv5_3)

"""

pool5_f = tf.reshape(pool2, [-1, 7*7*16])

fc1_w = gen_weight([7*7*16, 100])
fc1_b = gen_biase([100])
fc1 = tf.nn.relu(tf.matmul(pool5_f, fc1_w) + fc1_b)

# dropout
keep_prob = tf.placeholder(tf.float32)
fc1_drop = tf.nn.dropout(fc1, keep_prob)

fc2_w = gen_weight([100, 10])
fc2_b = gen_biase([10])
fc2 = tf.matmul(fc1_drop, fc2_w) + fc2_b

"""
fc2 = tf.nn.relu(tf.matmul(fc1_drop, fc2_w) + fc2_b)

# dropout
fc2_drop = tf.nn.dropout(fc2, keep_prob)

fc3_w = gen_weight([4096, 91])
fc3_b = gen_biase([91])
fc3 = tf.matmul(fc2_drop, fc3_w) + fc3_b
"""

pred = tf.nn.softmax(fc2)

lr = 0.0004
epoch_num = 20000
display_num = 100
batch_size = 100

figure_x, figure_y = [], []

cross_entropy = -tf.reduce_sum(y * tf.log(pred))
train_op = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

acc_num = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
acc_rate = tf.reduce_mean(tf.cast(acc_num, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

datasests = input_data.read_data_sets("./MNIST_data", one_hot=True)

for i in range(1, epoch_num+1):
    train_x, train_y = datasests.train.next_batch(batch_size)
    train_op.run(feed_dict={x: train_x, y: train_y, keep_prob: 0.5})
    if i % display_num == 0:
        acc = acc_rate.eval(feed_dict={x: train_x, y: train_y, keep_prob: 0.5})
        print("Step %d : acc_rate = %g" % (i, acc))
        figure_x.append(i)
        figure_y.append(acc)


test_acc_rate = acc_rate.eval(feed_dict={x: datasests.test.images, y: datasests.test.labels, keep_prob: 0.5})
print("Test acc is %g" % test_acc_rate)

plt.figure()
plt.xlabel("Epochs")
plt.ylabel("acc_rate")
plt.plot(figure_x, figure_y, "b--")
plt.show()
