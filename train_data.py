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
CACHE_DIR = 'D:/PythonCode/saved_model' #模型的储存目录
LEARNING_RATE = 0.1 #学习率
STEPS = 5000 #训练次数
METHOD_NUM = 3
#↑对图片调整大小的方法，其中：
#0.双线性插值法
#1.最近邻居法
#2.双三次插值法
#3.面积插值法


train_batch, train_label_batch = input_data.get_batch(input_data.create_image_lists(INPUT_DATA, True), IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
x = tf.placeholder(tf.float32, [None, IMG_W * IMG_H])
W = tf.Variable(tf.zeros([IMG_W * IMG_H, 34]), name = "weights" )
b = tf.Variable(tf.zeros([34]), name = "biases" )
y_without_softmax = tf.matmul(x, W) + b
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 34])
cross_entropy = tf.reduce_sum(tf.square(tf.subtract(y_,y)))
#cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1)) #这是另一种计算交叉熵的函数
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)



with tf.Session() as sess:
    tf.global_variables_initializer().run()
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    summer_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(CACHE_DIR, sess.graph)
    saver = tf.train.Saver()
    try:
        while not coord.should_stop() and i < STEPS:
            img, label = sess.run([train_batch, train_label_batch])

            train_step.run({x: np.reshape(img, [BATCH_SIZE, IMG_W * IMG_H]), y_: np.reshape(label, [BATCH_SIZE, 34])})
            
            if i % 100 == 0 :
                print("训练了" + str(i) + "次。")
                summer_str = sess.run(summer_op)
                print("预测："+str(sess.run(tf.argmax(sess.run(y, feed_dict = {x: np.reshape(img, [BATCH_SIZE, IMG_H * IMG_W])}), 1))))
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
