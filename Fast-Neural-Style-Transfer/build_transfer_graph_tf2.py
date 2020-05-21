#!/usr/bin/env python
# coding: utf-8

# ## 加载库
# In[1]:

# import tensorflow_addons as tfa # tensorflow2.0以上才能用
import numpy as np
import cv2 # pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ opencv-python
from imageio import imread, imsave
import scipy.io
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from img_preprocessing import resize_and_crop
from img_preprocessing import image_size


MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3)) # 通道颜色均值

# ## 构造转换网络，为卷积、残差、逆卷积结构

# In[10]:


# 这个cell 只需调用一次，重复调用会出错
# X = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='X')
def GetTransferGraph(tf, X):
    k_initializer = tf.truncated_normal_initializer(0, 0.1)

    def relu(x):
        return tf.nn.relu(x)

    def conv2d(inputs, filters, kernel_size = 3, strides = 1):
        p = int(kernel_size / 2)
        #填充边缘
        h0 = tf.pad(inputs, [[0, 0], [p, p], [p, p], [0, 0]], mode='reflect')
        #conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='valid', kernel_initializer=k_initializer)
        #return conv(inputs)
        return tf.layers.conv2d(inputs=h0, filters=filters, kernel_size=kernel_size, strides=strides, padding='valid', kernel_initializer=k_initializer)


    def deconv2d(inputs, filters, kernel_size = 3, strides = 1):
        shape = tf.shape(inputs)
        height, width = shape[1], shape[2]
        # 近邻插值法，
        h0 = tf.image.resize_images(inputs, [height * strides * 2, width * strides * 2], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return conv2d(h0, filters, kernel_size, strides)

    # 正则化
    # Batch_Norm是在一个Batch内不同样本在的标准化，而instance_norm在一个样本内的标准化。
    def instance_norm(inputs):
        return tf.contrib.layers.instance_norm(inputs)
        # return tfa.layers.InstanceNormalization(inputs)
    # 残差网络
    def residual(inputs, filters = 128, kernel_size = 3):
        h0 = relu(conv2d(inputs, filters, kernel_size, 1))
        h0 = conv2d(h0, filters, kernel_size, 1)
        return tf.add(inputs, h0)


    with tf.variable_scope('transformer', reuse=None):
        h0 = tf.pad(X - MEAN_VALUES, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='reflect')
        h0 = relu(instance_norm(conv2d(h0, 32, 9, 1)))
        h0 = relu(instance_norm(conv2d(h0, 64, 3, 2)))
        h0 = relu(instance_norm(conv2d(h0, 128, 3, 2)))

        for i in range(5):
            h0 = residual(h0, 128, 3)

        h0 = relu(instance_norm(deconv2d(h0, 64, 3, 2)))
        h0 = relu(instance_norm(deconv2d(h0, 32, 3, 2)))
        h0 = tf.nn.tanh(instance_norm(conv2d(h0, 3, 9, 1)))
        h0 = (h0 + 1) / 2 * 255.
        shape = tf.shape(h0)
        g = tf.slice(h0, [0, 10, 10, 0], [-1, shape[1] - 20, shape[2] - 20, -1], name='generate')
        return g

