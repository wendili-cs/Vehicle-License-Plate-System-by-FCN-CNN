'''
TensorFlow 1.3
Python 3.6
By LiWenDi
'''
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

CAPACITY = 500 #每次随机批次的总数量
IMG_H = 40 #图像的高
IMG_W = 20 #图像的宽
METHOD_NUM = 3
#↑对图片调整大小的方法，其中：
#0.双线性插值法
#1.最近邻居法
#2.双三次插值法
#3.面积插值法




def create_image_lists(input_data, if_shuffled):
    result = {}
    sub_dirs = [x[0] for x in os.walk(input_data)]
    is_root_dir = True
    image_labels = []
    file_list = {}
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        image_labels.append(os.path.basename(sub_dir))
        image_dirs =  os.listdir(sub_dir)
        real_image_dirs = []
        for allDir in image_dirs:
            allDir = os.path.join('%s/%s' % (sub_dir, allDir))
            real_image_dirs.append(allDir)
        file_list[os.path.basename(sub_dir)] = real_image_dirs

    if if_shuffled:
        for key in file_list:
            random.shuffle(file_list[key])
    #到此是读取所有文件夹里面的文件并且分类到各自文件夹的字典之中
    
    return file_list

'''
def file_list_to_imgs_and_labels(file_list):
    for key in file_list:
'''

def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path



def get_batch(image_lists, image_H, image_W, batch_size, capacity):
    
    image = []
    label = []

    for key in image_lists:
        for elements in image_lists[key]:
            image.append(elements)
            label.append(key)

    label_machine = []
    for each_label in label:
        temp = [0] * 34
        if ord(each_label) >= 48 and ord(each_label) <= 57:
            temp[ord(each_label) - 48] = 1
        elif ord(each_label) >= 65 and ord(each_label) < 73:
            temp[ord(each_label) - 55] = 1
        elif ord(each_label) >= 74 and ord(each_label) < 79:
            temp[ord(each_label) - 56] = 1
        elif ord(each_label) >= 80 and ord(each_label) <= 90:
            temp[ord(each_label) - 57] = 1
        else:
            print("分类错误！")
        label_machine.append(temp)
    
    input_queue = tf.train.slice_input_producer([image, label_machine])

    label_machine = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_contents, channels = 1)
    image = tf.image.resize_images(image, [image_W, image_H], method= METHOD_NUM)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label_machine], batch_size = batch_size, num_threads = 64, capacity = capacity)
    label_batch = tf.reshape(label_batch, [batch_size, 34])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch
