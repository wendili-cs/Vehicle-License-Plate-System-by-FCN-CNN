import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pylab
import os

IMG_H = 40
IMG_W = 20
METHOD_NUM = 3
#↑对图片调整大小的方法，其中：
#0.双线性插值法
#1.最近邻居法
#2.双三次插值法
#3.面积插值法

#↓模型读取路径
CACHE_DIR = 'D:\PythonCode\saved_model_cnn'

#↓文件输入路径
INPUT_PIC = "car_pic_test/02.png"

def Identification(input_image):

    with tf.Graph().as_default():
        sess2 = tf.Session()
        BATCH_SIZE = 1 #这里只预测一个图片所以是1
        N_CLASSES = 34 #34个预测结果

        '''
        image_show = Image.open(INPUT_PIC)
        image_contents = tf.read_file(INPUT_PIC)
        image = tf.image.decode_png(image_contents, channels = 1)
        image = tf.image.resize_images(image, [IMG_H, IMG_W], method= METHOD_NUM)
        image = tf.image.per_image_standardization(image)
        image = tf.cast(image, tf.float32)
        image = sess2.run(image)
        image = np.reshape(image, [BATCH_SIZE, IMG_W* IMG_H])
        '''
        input_image_temp = []
        for each_H in input_image:
            temp = []
            for each_W in each_H:
                temp.append(each_W)
            input_image_temp.append(temp)
        input_image_temp = np.resize(input_image_temp, [ len(input_image_temp), len(input_image_temp[0]),BATCH_SIZE])
        input_image_temp = tf.image.resize_images(input_image_temp, [IMG_H, IMG_W], method= METHOD_NUM)
        input_image_temp = tf.image.per_image_standardization(input_image_temp)
        input_image_temp = tf.cast(input_image_temp, tf.float32)
        input_image_temp = sess2.run(input_image_temp)
        image = np.reshape(input_image_temp, [-1, IMG_H, IMG_W, 1])

        x_image = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, 1])
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
        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices = [1]))
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1)) #这是另一种计算交叉熵的函数
        #train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            #print("读取数据...")
            saver.restore(sess, os.path.join(CACHE_DIR, 'model.ckpt'))
            prediction = sess.run(y_conv, feed_dict = {x_image: image, keep_prob: 1.0})
            #for each in prediction[0]:
                #print(format(each,'.1e'), end = " ")
            asc = sess.run(tf.argmax(prediction, 1))
            #print("这张图片的数字/字母为：")
            if asc[0] >= 0 and asc[0] <= 9:
                print(asc[0])
            elif asc[0] >=10 and asc[0] < 18:
                print(chr(asc[0] + 55))
            elif asc[0] >= 18 and asc[0] < 23:
                print(chr(asc[0] + 56))
            elif asc[0] >= 23 and asc[0] < 34:
                print(chr(asc[0] + 57))
            
            image = np.reshape(image, [IMG_H, IMG_W])
            plt.imshow(image)
            plt.show()

