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
CACHE_DIR = 'D:\PythonCode\saved_model'

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
                #print(each_W)
            input_image_temp.append(temp)
        #print(len(input_image_temp))
        #print(len(input_image_temp[0]))
        #print(len(input_image_temp[0][0]))
        input_image_temp = np.resize(input_image_temp, [ len(input_image_temp), len(input_image_temp[0]),BATCH_SIZE])
        #print(len(input_image_temp))
        #print(len(input_image_temp[0]))
        input_image_temp = tf.image.resize_images(input_image_temp, [IMG_H, IMG_W], method= METHOD_NUM)
        input_image_temp = tf.image.per_image_standardization(input_image_temp)
        input_image_temp = tf.cast(input_image_temp, tf.float32)
        input_image_temp = sess2.run(input_image_temp)
        #print(len(input_image_temp))
        #print(len(input_image_temp[0]))
        #print(len(input_image_temp[0][0]))
        image = np.reshape(input_image_temp, [BATCH_SIZE, IMG_W* IMG_H])

        W = tf.Variable(tf.zeros([IMG_H * IMG_W, 34]), name = "weights" )
        b = tf.Variable(tf.zeros([34]), name = "biases")

        saver = tf.train.Saver()
        with tf.Session() as sess:
            #print("读取数据...")
            saver.restore(sess, os.path.join(CACHE_DIR, 'model.ckpt'))
            prediction = sess.run(tf.nn.softmax(sess.run(tf.matmul(image, sess.run(W)) + sess.run(b))))
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
