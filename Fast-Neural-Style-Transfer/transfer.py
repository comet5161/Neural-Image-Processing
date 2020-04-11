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

tf.reset_default_graph()
X = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='X')
g = GetTransferGraph(tf, X)



def addPostfix(filename, postfix):
    ary = os.path.splitext(filename)
    if(len(ary) < 2):
        return filename + postfix
    return ary[0] + postfix + ary[1]

# In[]
# 
h_step = 800
w_step = 400
edge = 40 #边缘重叠的像素长度

def splitImage(img):
    units = []
    h = img.shape[0]
    w = img.shape[1]
    
    for i in range(0, h, h_step):
        if(i + edge >= h): 
            break
        up = max(i - edge, 0)
        down = min(i + h_step + edge, h)
        for j in range(0, w, w_step):
            if(j + edge >= w):
                break
            left = max(j - edge, 0)
            right = min(j + w_step + edge, w)
            unit = img[up : down, left : right, :]
            units.append(unit)
    return units
            

def mergeImage(units, shape):
    h = shape[0]
    w = shape[1]
    c = shape[2]
    img = np.ones(shape)
    idx = 0
    for i in range(0, h, h_step):
        if(i + edge >= h):
            break
        i_offset = edge
        if( i == 0):
            i_offset = 0
        i_end = min(h, i + h_step + edge)
        for j in range(0, w, w_step):
            if(j + edge >= w):
                break
            j_offset = edge
            if(j == 0):
                j_offset = 0
            j_end = min(w, j + w_step + edge)
            print("unit shape: " + str(units[idx].shape) )
            img[i:i_end, j:j_end, 0:c] = units[idx][i_offset:, j_offset:, :]

            #重叠部分平滑过渡
            if(j_offset > 0):
                print('j_offset')
                for k in range(edge):
                    p = k/(edge-1)
                    img[i:i_end, j - 1 - k, :] = img[i:i_end, j - 1 - k, :]*(p) + units[idx][i_offset:, j_offset - 1 - k, : ]*(1-p)
            if(i_offset > 0):
                for k in range(edge):
                    p = k/(edge-1)
                    img[i - 1 - k, j:j_end, :] = img[i - 1 - k, j : j_end, :]*(p) + units[idx][i - 1 - k:, j_offset :, : ]*(1-p)

            idx += 1
    return img

# # Test
# a = imread('content/fudan.png')
# # a = imread('content/fudan_style_tree.png')
# b = splitImage(a)

# k = 0
# for i in b :
#     k += 1
#     imsave("content/fudan_unit" + str(k) + ".png", i)

# c = mergeImage(b, a.shape)
# imsave("content/fudan__merge.png", c)


# In[1]:
def beginTransfer(content_img_path =  'content/fudan.png', transfer_img_path = "", style_id = 0, device = "gpu", with_src = False):
    print("start transfer image: " + content_img_path)
    #不使用gpu
    if(device == "cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    #content_img_path = 'content/fudan.jpg'

    # ## 查看风格图片

    # In[]
    style_models = glob.glob('models/style_*')
    model_dir = style_models[style_id]
    model_name = os.path.basename(model_dir)
    if(transfer_img_path == ""):
        transfer_img_path = addPostfix(content_img_path, "_"+model_name)

    X_input= imread(content_img_path)

    X_sample = X_input[:, :, 0:3] # 只保留３通道（png图片有４通道）
    h_sample = X_sample.shape[0]
    w_sample = X_sample.shape[1]

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_run_opt = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    tf_config.gpu_options.allow_growth=True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6

    

    sess = tf.Session(config = tf_config)
    saver = tf.train.Saver()
    print("load model.")
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    writer = tf.summary.FileWriter('transfer/', sess.graph)

    # saver = tf.train.import_meta_graph(model_dir + 'trained_model.meta')
    # # 方法一：加载指定模型数据
    # #saver.restore(sess,  model_dir + 'trained_model')
    # # 方法二：加载最近保存的数据
    # saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    # g = sess.graph.get_tensor_by_name('transformer/generate:0')
    # X = sess.graph.get_tensor_by_name('X:0')

    def unitTransfer(unit):
        h = unit.shape[0]
        w = unit.shape[1]
        gen_img = sess.run(g, feed_dict={X: [unit]}, options = tf_run_opt)[0]
        gen_img = np.clip(gen_img, 0, 255)
        if(with_src == True):
            result = np.zeros((h, w * 2, 3))
            result[:, :w, :] = unit / 255.
            result[:, w:, :] = gen_img[:h, :w, :] / 255.
        else:
            result = np.zeros((h, w, 3))
            result[:h, :w, :] = gen_img[:h, :w, :] / 255.
        return result

    print("begin transfer")
    shape = X_sample.shape
    units = splitImage(X_sample)
    for i in range(len(units)):
        units[i] = unitTransfer(units[i])
    generate_img = mergeImage(units, shape)
        
    # plt.axis('off')
    # plt.imshow(result)
    # plt.show()
    imsave(transfer_img_path, generate_img)

    writer.close()

    return transfer_img_path








# %%
#beginTransfer()