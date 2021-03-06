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
import pandas

#不使用gpu
os.environ["CUDA_VISIBLE_DEVICES"] = ""

content_img_path = 'content/fudan.png'


# ## 查看风格图片
style_id = 0
style_images = glob.glob('styles/*.jpg')
style_img_path = style_images[style_id]
style_name = style_img_path[style_img_path.find('/') + 1:].rstrip('.jpg')
model_dir = 'models/style_%s/' % style_name

transfer_img_path = content_img_path.rstrip('.png') + '_style_' + style_name + '.png'

X_sample = imread(content_img_path)
X_sample = X_sample[:, :, 0:3] # 只保留３通道（png图片有４通道）
h_sample = X_sample.shape[0]
w_sample = X_sample.shape[1]

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth=True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6

sess  =  tf.Session(config = tf_config)
saver = tf.train.import_meta_graph('models/transfer_model.meta')
# 方法一：加载指定模型数据
#saver.restore(sess,  model_path + 'trained_model')
# 方法二：加载最近保存的数据
saver.restore(sess, tf.train.latest_checkpoint(model_dir))
g = sess.graph.get_tensor_by_name('transformer/g:0')
X = sess.graph.get_tensor_by_name('X:0')

# In[]

vars = tf.global_variables()

# In[]
# for var in vars:
#     print(var)

tensors = tf.get_default_graph().as_graph_def().node

# In[]
for ts in tensors[0:10]:
    print(ts.name)





# %%

writer = tf.summary.FileWriter('models/', sess.graph)
writer.close()

# %%
