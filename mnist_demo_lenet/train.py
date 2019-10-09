from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pylab as plt


def gen_conv(input, weight, padding):
    return tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding=padding)


def gen_pool2x2(input, padding):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)


def gen_weight(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=0.1))


def gen_biase(shape):
    return tf.Variable(tf.constant(shape=shape, value=0.1))


datasets = input_data.read_data_sets("./MNIST_data", one_hot=True)
print(datasets.train.images.shape)

x = tf.placeholder(tf.float32, [None, 28*28], name="input")
x_ = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

#28*28@1
conv1_w = gen_weight([5, 5, 1, 6])
conv1_b = gen_biase([6])
conv1 = tf.nn.relu(gen_conv(x_, conv1_w, "SAME") + conv1_b)

#28*28@6
pool1 = gen_pool2x2(conv1, "SAME")

#14*14@6
conv2_w = gen_weight([5, 5, 6, 16])
conv2_b = gen_biase([16])
conv2 = tf.nn.relu(gen_conv(pool1, conv2_w, "VALID") + conv2_b)

#10*10@16
pool2 = gen_pool2x2(conv2, "SAME")

#5*5@16
conv3_w = gen_weight([5, 5, 16, 120])
conv3_b = gen_biase([120])
conv3 = tf.nn.relu(gen_conv(pool2, conv3_w, "VALID") + conv3_b)

#1*1@120
h_conv = tf.reshape(conv3, [-1, 1*1*120])
fc1_w = gen_weight([120, 84])
fc1_b = gen_biase([84])
fc1 = tf.nn.relu(tf.matmul(h_conv, fc1_w) + fc1_b)

#1*1@84
fc2_w = gen_weight([84, 10])
fc2_b = gen_biase([10])
pred = tf.nn.softmax(tf.matmul(fc1, fc2_w) + fc2_b, name="predict")

#1*1@10


l_r = 1e-4
epochs = 20000
batch_size = 50
display_num = 1000

figure_x, figure_y = [], []

cross_entropy = -tf.reduce_sum(y*tf.log(pred))
train_op = tf.train.AdamOptimizer(l_r).minimize(cross_entropy)

acc_num = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
acc_rate_op = tf.reduce_mean(tf.cast(acc_num, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

for i in range(1, epochs+1):
    batch = datasets.train.next_batch(batch_size, shuffle=True)
    # print(batch)
    train_op.run(feed_dict={x: batch[0], y: batch[1]})
    if i % display_num == 0:
        acc_rate = acc_rate_op.eval(feed_dict={x: batch[0], y: batch[1]})
        figure_x.append(i)
        figure_y.append(acc_rate)
        print("step %d : acc_rate is %g " % (i, acc_rate))

saver.save(sess, "./checkpoint_dir/mnist_model_%d" % epochs)

test_acc_rate = acc_rate_op.eval(feed_dict={x: datasets.test.images, y: datasets.test.labels})
print("Test result: acc_rate is %g " % test_acc_rate)

plt.figure()
plt.xlabel("epochs")
plt.ylabel("acc_rate")
plt.plot(figure_x, figure_y, "b--")
plt.axis([1, epochs, 0, 1])
plt.show()