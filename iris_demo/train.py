import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import pandas

def get_data(path):
    data = np.asarray(pandas.read_csv(path))

    d1 = data[data[:, -1] == 'setosa'][:, 1:]
    d2 = data[data[:, -1] == 'versicolor'][:, 1:]
    d3 = data[data[:, -1] == 'virginica'][:, 1:]

    d1[:, -1], d2[:, -1], d3[:, -1] = 0, 1, 2

    data = np.vstack((d1, d2, d3))

    label = np.zeros((data.shape[0], 3))
    for i in range(data.shape[0]):
        label[i, data[i,-1]] = 1
    data = np.hstack((data[:, :-1], label))

    return data


def getW(shape):
    return tf.Variable(tf.truncated_normal(shape, 0, 0.1))


def getB(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def layer( inputLayer, input_units, units, activation="relu"):
    W = getW([input_units, units])
    B = getB([units])
    if(activation == "tanh"):
        return tf.nn.tanh(tf.add(tf.matmul(inputLayer, W), B))
    elif(activation == "elu"):
        return tf.nn.elu(tf.add(tf.matmul(inputLayer, W), B))
    elif(activation == "softmax"):
        return tf.nn.softmax(tf.matmul(inputLayer, W) + B)
    else:
        return tf.nn.relu(tf.add(tf.matmul(inputLayer, W), B))


def split_train_test( data, split_rate=0.8):
    n = data.shape[0]
    train_num = (int)(n*split_rate)
    print(train_num)
    return data[:train_num, :], data[train_num:, :]


data = get_data('./iris.csv')
np.random.shuffle(data)
train, test = split_train_test(data, split_rate=0.9)

l_r = 0.0004
num_input = 4
num_calsses = 3
n_hidden_1 = 12
n_hidden_2 = 12
n_hidden_3 = 12
batch_size = 50
display_num = 20

x = tf.placeholder(tf.float32, [None, num_input])
y = tf.placeholder(tf.float32, [None, num_calsses])


layer_1 = layer(x, num_input, n_hidden_1)
layer_2 = layer(layer_1, n_hidden_1, n_hidden_2, "relu")
layer_3 = layer(layer_2, n_hidden_2, n_hidden_3, "relu")
output_layer = layer(layer_3, n_hidden_3, num_calsses, "softmax")


loss_op = -tf.reduce_sum(y * tf.log(output_layer))
train_op = tf.train.AdamOptimizer(l_r).minimize(loss_op)

correct_num = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
acc_rate = tf.reduce_mean(tf.cast(correct_num, tf.float32))

figure_x = []
figure_y = []

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
epochs = 4000
for i in range(1, epochs+1):
    np.random.shuffle(train)
    x_batch, y_batch = train[: batch_size, :-3], train[: batch_size, -3:]
    train_op.run(feed_dict={x: x_batch, y: y_batch})

    if i % display_num == 0:
        train_acc = acc_rate.eval(feed_dict={x: x_batch, y: y_batch})
        figure_x.append(i)
        figure_y.append(train_acc)
        print("step %d, train_acc is %f" % (i, train_acc))

test_acc = acc_rate.eval(feed_dict={x: test[:, :-3], y: test[:, -3:]})
print("test_acc is %f " % test_acc)


plt.figure()
plt.xlabel("Epochs")
plt.ylabel("acc_rate")
plt.plot(figure_x, figure_y, "b--", linewidth=1)
plt.show()