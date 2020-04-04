#!/usr/bin/env python
# coding: utf-8

# ## 加载库

# In[1]:


# -*- coding: utf-8 -*-


import numpy as np
#import cv2 # pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ opencv-python
from cv2 import cv2 as cv
from imageio import imread, imsave
import scipy.io
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')



# ## 加载图格图片

# In[3]:

def resize_and_crop(image, image_size):
    h = image.shape[0]
    w = image.shape[1]
    if h > w:
        image = image[h//2 - w//2 : h//2 + w//2, : , :]
    elif h < w:
        image = image[: , w//2 - h//2 : w//2 + h//2, :]
    image = cv.resize(image, (image_size, image_size))
    return image

X_data = []
image_size = 256
paths = glob.glob('content/train2014_5000/*.jpg')

for i in tqdm(range(len(paths))):
    image = imread(paths[i])
    if len(image.shape) < 3: # 筛掉黑白图片
        continue
    X_data.append(resize_and_crop(image, image_size))
#plt.imshow(X_data[-1])
X_data = np.array(X_data)
#print(X_data.shape)
np.save('content/train2014_5000.preprocessing.npy', X_data)

