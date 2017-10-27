'''
TensorFlow 1.3
Python 3.6
By LiWenDi
'''
import tensorflow as tf
import input_data
import os
import numpy as np
import matplotlib.pyplot as plt

CHANNELS = 3 #色彩读取通道
BATCH_SIZE = 10 #训练批次的数量
CAPACITY = 500 #每次随机批次的总数量
IMG_H = 40 #图像的高
IMG_W = 20 #图像的宽
INPUT_DATA = "charSamples/" #训练数据的根目录
CACHE_DIR = 'D:/PythonCode/saved_model_cnn' #模型的储存目录
LEARNING_RATE = 0.0001 #学习率
STEPS = 5000 #训练次数
METHOD_NUM = 3
#↑对图片调整大小的方法，其中：
#0.双线性插值法
#1.最近邻居法
#2.双三次插值法
#3.面积插值法


train_batch, train_label_batch = input_data.get_batch(input_data.create_image_lists(INPUT_DATA, True), IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
x = tf.placeholder(tf.float32, [None, IMG_W * IMG_H])
x_image = tf.reshape(x, [-1, IMG_H, IMG_W, 1])
#W = tf.Variable(tf.zeros([IMG_W * IMG_H, 34]), name = "weights" )
#b = tf.Variable(tf.zeros([34]), name = "biases" )

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name = name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
b_conv1 = bias_variable([32], 'b_conv1')
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
b_conv2 = bias_variable([64], 'b_conv2')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([IMG_H*IMG_W*4, 1024], 'W_fc1')
b_fc1 = bias_variable([1024], 'b_fc1')
h_pool2_flat = tf.reshape(h_pool2, [-1, IMG_H*IMG_W*4])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 34], 'W_fc2')
b_fc2 = bias_variable([34], 'b_fc2')
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#y_without_softmax = tf.matmul(x, W) + b
#y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 34])
#cross_entropy = tf.reduce_sum(tf.square(tf.subtract(y_,y_conv)))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices = [1]))
#cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1)) #这是另一种计算交叉熵的函数
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)



with tf.Session() as sess:
    tf.global_variables_initializer().run()
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    summer_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(CACHE_DIR, sess.graph)
    saver = tf.train.Saver()
    print(h_conv1)
    print(h_conv2)
    print(h_pool2)
    try:
        while not coord.should_stop() and i < STEPS:
            img, label = sess.run([train_batch, train_label_batch])

            train_step.run({x: np.reshape(img, [BATCH_SIZE, IMG_W * IMG_H]), y_: np.reshape(label, [BATCH_SIZE, 34]), keep_prob: 0.5})
            
            if i % 100 == 0 :
                print("训练了" + str(i) + "次。")
                summer_str = sess.run(summer_op)
                print("预测："+str(sess.run(tf.argmax(sess.run(y_conv, feed_dict = {x: np.reshape(img, [BATCH_SIZE, IMG_H * IMG_W]), keep_prob: 1.0}), 1))))
                print("标签："+str(sess.run(tf.argmax(np.reshape(label, [BATCH_SIZE, 34]), 1))))
                
                '''
                for each_img in np.reshape(img, [BATCH_SIZE, IMG_H , IMG_W, 1]):
                    each_img = np.reshape(each_img, [IMG_H , IMG_W])
                    plt.imshow(each_img)
                    plt.show()
                #取消注释可以展示标签对应的图片
                '''

            if i == STEPS - 1:
                print("一切完成！")
                checkpoint_path = os.path.join(CACHE_DIR, 'model.ckpt')
                saver_path = saver.save(sess, checkpoint_path)
            i += 1
            

    except tf.errors.OutOfRangeError:
        print("完成！")
    finally:
        coord.request_stop()
    coord.join(threads)
