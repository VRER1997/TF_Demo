import tensorflow as tf
import numpy as np
import cv2
import os

epoch_num = 20000

saver = tf.train.import_meta_graph("./checkpoint_dir/mnist_model_%d.meta" % epoch_num)

sess = tf.InteractiveSession()

saver.restore(sess, tf.train.latest_checkpoint("./checkpoint_dir"))

graph = tf.get_default_graph()

varible_names = [v.name for v in graph.get_operations()]

predict_op = graph.get_tensor_by_name("predict:0")
x = graph.get_tensor_by_name("input:0")

cv2.namedWindow("[Picture]")
pic_count = 500

basepath = "./mnist_train_pic/"
filenames = os.listdir(basepath)

for file in filenames:
    img = cv2.imread(basepath + file, cv2.IMREAD_GRAYSCALE)

    array = np.array(img)
    res_array = sess.run(predict_op, feed_dict={x: array.reshape(1, 784)})[0].tolist()
    predict_result = res_array.index(max(res_array))
    cv2.imshow("[Picture]", img)
    print("Result: {0}".format(predict_result))
    cv2.waitKey(0)


