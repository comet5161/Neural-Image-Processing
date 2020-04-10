#!/usr/bin/env python
# coding: utf-8

# ## 加载库
# In[1]:
import tensorflow as tf
# import tensorflow_addons as tfa # tensorflow2.0以上才能用
import numpy as np
import cv2 # pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ opencv-python
from imageio import imread, imsave
import scipy.io
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from build_transfer_graph import GetTransferGraph

def addPostfix(filename, postfix):
    ary = os.path.splitext(filename)
    if(len(ary) < 2):
        return filename + postfix
    return ary[0] + postfix + ary[1]


def beginTransfer(content_img_path =  'content/fudan.png', transfer_img_path = "", style_id = 0, with_src = False):
    print("start transfer image: " + content_img_path)
    #不使用gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    #content_img_path = 'content/fudan.jpg'

    # ## 查看风格图片

    # In[]
    style_models = glob.glob('models/style_*')
    model_dir = style_models[style_id]
    model_name = os.path.basename(model_dir)
    if(transfer_img_path == ""):
        transfer_img_path = addPostfix(content_img_path, model_name)

    X_input= imread(content_img_path)

    X_sample = X_input[:, :, 0:3] # 只保留３通道（png图片有４通道）
    h_sample = X_sample.shape[0]
    w_sample = X_sample.shape[1]

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth=True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6

    X = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='X')
    g = GetTransferGraph(tf, X)

    with tf.Session(config = tf_config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        # saver = tf.train.import_meta_graph(model_dir + 'trained_model.meta')
        # # 方法一：加载指定模型数据
        # #saver.restore(sess,  model_dir + 'trained_model')
        # # 方法二：加载最近保存的数据
        # saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        # g = sess.graph.get_tensor_by_name('transformer/generate:0')
        # X = sess.graph.get_tensor_by_name('X:0')
        
        gen_img = sess.run(g, feed_dict={X: [X_sample]})[0]
        gen_img = np.clip(gen_img, 0, 255)
        if(with_src == True):
            result = np.zeros((h_sample, w_sample * 2, 3))
            result[:, :w_sample, :] = X_sample / 255.
            result[:, w_sample:, :] = gen_img[:h_sample, :w_sample, :] / 255.
        else:
            result = np.zeros((h_sample, w_sample, 3))
            result[:h_sample, :w_sample, :] = gen_img[:h_sample, :w_sample, :] / 255.
        # plt.axis('off')
        # plt.imshow(result)
        # plt.show()
        imsave(transfer_img_path, result)
        return transfer_img_path







