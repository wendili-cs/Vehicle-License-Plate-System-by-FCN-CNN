'''
TensorFlow 1.3
Python 3.6
By LiWenDi
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import tryOne_def
import tryOne_def_cnn


USE_CNN = True
IMG_W = 720
IMG_H = 200
INPUT = "car_pic_test/1.png"
JUDGE = 4000
JUDGE_EACH = 5000 #划分单个字母数字的阈值
LETTER_W = 30 #最短字母宽
SEC_LETTER_W = 15
def devide_plate(input_dir):
    sess = tf.Session()
    img_for_show = cv2.imread(input_dir)
    cv2.imshow('Image', img_for_show)
    cv2.waitKey(0)
    image_content = tf.read_file(input_dir)
    image = tf.image.decode_png(image_content, channels = 1)
    image = tf.image.resize_images(image, [IMG_H, IMG_W])
    image = tf.cast(image, tf.float32)
    image_data = sess.run(image)
    image_data_show = np.reshape(image_data, [IMG_H, IMG_W])
    retval, image_data = cv2.threshold(image_data, 180, 255, cv2.THRESH_BINARY)
    #image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    #plt.imshow(sess.run(image_content))#, cmap = 'Greys'
    #plt.show()
    cv2.imshow('Image', image_data)
    cv2.waitKey(0)
    #print(len(image_data))
    lines_sum = []
    column_sum = []
    #print(len())
    for i in range(IMG_W):
        column_sum.append(np.sum(image_data[20:180,i]))
        #print(column_sum[i])
    '''
    cut_up = cut_down = IMG_H/2
    do_cut_up = do_cut_down = False
    for i in range(IMG_H/2 - 1):
        if lines_sum[cut_up] - lines_sum[cut_up-1] < JUDGE and not do_cut_up:
            do_cut_up = True
            cut_up -= 1
        elif not do_cut_up:
            cut_up -= 1

        if lines_sum[cut_down] - lines_sum[cut_down+1] < JUDGE and not do_cut_down:
            do_cut_down = True
            cut_down -= 1
        elif not do_cut_up:
            cut_down -= 1
    '''
    cut_point = 0
    cut_left = cut_right = IMG_W/2
    do_cut_left = do_cut_right = False
    letters = []
    letter_len = 0
    cutting = False
    #through_first = False
    has_put = False
    #for each in column_sum:
        #print(each)
    for i in range(10,IMG_W-10):
        
        if column_sum[i] > JUDGE_EACH and cutting:
            letter_len += 1
        elif column_sum[i] > JUDGE_EACH:
            cutting = True
            cut_point = i
            letter_len = 1
            #print("开始切")
        elif cutting:
            if letter_len > LETTER_W:
                letters.append(image_data[15:IMG_H-7,cut_point - 5:(cut_point + letter_len + 5 )])
                cutting = False
                letter_len = 0
            #else:

    print("这个车牌号为：")
    discard_one = False
    for each in letters:
        if not discard_one:
            discard_one = True
        else:
        #cv2.imshow('Image', each)
        #cv2.waitKey(0)
            if USE_CNN:
                tryOne_def_cnn.Identification(each)
            else:
                tryOne_def.Identification(each)

    #if i in range(IMG_W/2 - 1):

    
    #for each_line in image_data:


devide_plate(INPUT)
