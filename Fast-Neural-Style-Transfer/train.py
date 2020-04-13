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
from img_preprocessing import resize_and_crop
from img_preprocessing import image_size
from build_transfer_graph import GetTransferGraph
#get_ipython().run_line_magic('matplotlib', 'inline')

import sys, getopt

def main(argv):
    batch_size = 1 # 原本是４，但显存会不够。
    epochs = 3
    max_train_samples = 1000*1000*1000
    write_log = False
    # ## 查看风格图片
    style_images = glob.glob('styles/*.jpg')
    style_img_path = style_images[2]
    style_img_path = 'styles/tree.jpg'
    #style_img_path = 'styles/wave.jpg'

    #解析命令行参数
    try:
        opts, args = getopt.getopt(argv, '', ['style_img=', 'epochs=', 'batch_size=', 'max_train_samples='])
    except getopt.GetoptError:
        print('train.py style_img=/path/to/img.jpg epochs=3 batch_size=1 ')
        sys.exit(2)
    for opt, arg in opts:
        if(opt == '--style_img'):
            if(os.path.exists(arg)):
                style_img_path = arg
            else:
                print('file ' + arg + 'does not exists.')
                sys.exit(2);
        elif(opt == '--epochs'):
            epochs = int(arg)
        elif(opt == '--batch_size'):
            batch_size = int(arg)
        elif(opt == '--max_train_samples'):
            max_train_samples = int(arg)
        elif(opt == '--write_log'):
            write_log = True


    # ## 加载图片
    X_data = np.load('train/train2014_5000.preprocessing.npy')

    # # 加载vgg19模型，将模型中的参数设为常量
    vgg = scipy.io.loadmat('train/imagenet-vgg-verydeep-19.mat')
    vgg_layers = vgg['layers']
    #print(vgg_layers.shape)

    def vgg_endpoints(inputs, reuse=None):
        with tf.variable_scope('vgg_19', reuse=reuse): #模型中的变量都在变量空间vgg_endpoints里
            def _weights(layer, expected_layer_name):
                W = vgg_layers[0][layer][0][0][2][0][0]
                b = vgg_layers[0][layer][0][0][2][0][1]
                layer_name = vgg_layers[0][layer][0][0][0][0]
                assert layer_name == expected_layer_name
                return W, b

            def _conv2d_relu(prev_layer, layer, layer_name):
                W, b = _weights(layer, layer_name)
                W = tf.constant(W)
                b = tf.constant(np.reshape(b, (b.size)))
                return tf.nn.relu(tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b)

            def _avgpool(prev_layer):
                return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            graph = {}
            graph['conv1_1']  = _conv2d_relu(inputs, 0, 'conv1_1')
            graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
            graph['avgpool1'] = _avgpool(graph['conv1_2'])
            graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
            graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
            graph['avgpool2'] = _avgpool(graph['conv2_2'])
            graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
            graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
            graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
            graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
            graph['avgpool3'] = _avgpool(graph['conv3_4'])
            graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
            graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
            graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
            graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
            graph['avgpool4'] = _avgpool(graph['conv4_4'])
            graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
            graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
            graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
            graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
            graph['avgpool5'] = _avgpool(graph['conv5_4'])
            return graph


    # ## 将风格图片输入vgg19网络中，得到4个风格层的激活值，计算各风格层的Gram矩阵。

    X_style_data = resize_and_crop(imread(style_img_path), image_size)
    X_style_data = np.expand_dims(X_style_data, 0)
    print(X_style_data.shape)

    MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3)) # 通道颜色均值

    with tf.variable_scope('Style_input'):
        X_style = tf.placeholder(dtype=tf.float32, shape=X_style_data.shape, name='X_style')
        
    with tf.variable_scope('style_vgg'):
        style_endpoints = vgg_endpoints(X_style - MEAN_VALUES)
    STYLE_LAYERS = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']
    style_features = {}

    # tensorflow 用gpu训练时，默认占用所有gpu的显存，
    # 
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth=True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6

    sess = tf.Session(config = tf_config)

    for layer_name in STYLE_LAYERS:
        features = sess.run(style_endpoints[layer_name], feed_dict={X_style: X_style_data})
        features = np.reshape(features, (-1, features.shape[3]))
        gram = np.matmul(features.T, features) / features.size
        print(features.shape, gram.shape)
        style_features[layer_name] = gram


    # ## 构造转换网络，为卷积、残差、逆卷积结构

    # In[10]:
    with tf.variable_scope('Content_inputs'):
        X = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='X')
    #  只需调用一次，重复调用会出错
    g = GetTransferGraph(tf, X)

    # ## 将迁移图片和原始图片输入到 vgg19，得到各自对应输出
    # ### 计算内容损失

    CONTENT_LAYER = 'conv3_3'
    with tf.variable_scope('content_vgg'):
        content_endpoints = vgg_endpoints(X - MEAN_VALUES, True)
    with tf.variable_scope('generate_vgg'):
        g_endpoints = vgg_endpoints(g - MEAN_VALUES, True)

    def get_content_loss(endpoints_x, endpoints_y, layer_name):
        x = endpoints_x[layer_name]
        y = endpoints_y[layer_name]
        return 2 * tf.nn.l2_loss(x-y) / tf.to_float(tf.size(x))

    with tf.variable_scope('Content_loss'):
        content_loss = get_content_loss(content_endpoints, g_endpoints, CONTENT_LAYER)
        tf.identity(content_loss, name='content_loss')


    # ## 计算风格损失

    style_loss = []
    with tf.variable_scope('Style_loss'):
        for layer_name in STYLE_LAYERS:
            layer = g_endpoints[layer_name]
            shape = tf.shape(layer)
            bs, height, width, channel = shape[0], shape[1], shape[2], shape[3]
            
            features = tf.reshape(layer, (bs, height * width, channel))
            gram = tf.matmul(tf.transpose(features, (0,2,1)), features) / tf.to_float(height * width * channel)
            
            style_gram = style_features[layer_name]
            style_loss.append(2 * tf.nn.l2_loss(gram - style_gram) / tf.to_float(tf.size(layer)))
        style_loss = tf.reduce_sum(style_loss, name = 'style_loss')


    # ## 计算全变差正则， 得到总损失函数

    def get_total_variation_loss(inputs):
        h = inputs[:, :-1, :, :] - inputs[:, 1:, :, :]
        w = inputs[:, :, 1:, :]
        return tf.nn.l2_loss(h) / tf.to_float(tf.size(h)) + tf.nn.l2_loss(w) / tf.to_float(tf.size(w))

    with tf.variable_scope('Variation_loss'):
        total_variation_loss = get_total_variation_loss(g)
        tf.identity(total_variation_loss, name='variation_loss')

    content_weight = 1
    style_weight = 250
    total_variation_weight = 0.01
    with tf.variable_scope('Total_loss'):
        weighted_content_loss = tf.identity( content_weight * content_loss, name='weighted_content_loss')
        weighted_style_loss = tf.identity(style_weight * style_loss, name='weighted_style_loss')
        weighted_variation_loss = tf.identity(total_variation_weight * total_variation_loss, name='weighted_variation_loss')
        loss =  weighted_content_loss + weighted_style_loss + weighted_variation_loss
        tf.identity(loss, 'total_loss')


    # ## 定义优化器
    # In[16]:
    vars_t = [var for var in tf.trainable_variables() if var.name.startswith('transformer')]
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, var_list=vars_t)

    # ## 训练模型
    # In[20]:

    #style_name = style_img_path[style_img_path.find('/') + 1:].rstrip('.jpg')
    style_name = os.path.basename(style_img_path)
    style_name = os.path.splitext(style_name)[0]
    OUTPUT_DIR = 'models/style_%s' % style_name
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    tf.summary.scalar('losses/content_loss', content_loss)
    tf.summary.scalar('losses/style_loss', style_loss)
    tf.summary.scalar('losses/total_variation_loss', total_variation_loss)
    tf.summary.scalar('losses/loss', loss)
    tf.summary.scalar('weighted_losses/weighted_content_loss', weighted_content_loss)
    tf.summary.scalar('weighted_losses/weighted_style_loss', weighted_style_loss)
    tf.summary.scalar('weighted_losses/weighted_total_variation_loss', weighted_variation_loss)
    tf.summary.image('transformed', g)
    tf.summary.image('origin', X)
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(OUTPUT_DIR)

    sess.run(tf.global_variables_initializer())
    losses = []
    

    X_sample = imread('content/0.jpg')
    h_sample = X_sample.shape[0]
    w_sample = X_sample.shape[1]

    # In[]

    if(write_log):
        writer = tf.summary.FileWriter('train/', sess.graph)
    
    # tf_vars = tf.global_variables()
    # var_to_save = [var for var in tf_vars if 'transformer' in var.name]
    # saver = tf.train.Saver(var_to_save)
    saver = tf.train.Saver()
    #saver.save(sess, 'models/transfer_model')
    for e in range(epochs):
        data_index = np.arange(X_data.shape[0])
        np.random.shuffle(data_index)
        X_data = X_data[data_index]
        
        #for i in tqdm(range(X_data.shape[0] // batch_size)):
        train_range = range(X_data.shape[0] // batch_size)
        if( max_train_samples < X_data.shape[0]):
            train_range = range(max_train_samples //batch_size)
        #train_range = range(10)
        for i in tqdm(train_range):
            X_batch = X_data[i * batch_size: i * batch_size + batch_size]
            run_option = tf.RunOptions()
            run_option.report_tensor_allocations_upon_oom = True
            ls_, _ = sess.run([loss, optimizer], feed_dict={X: X_batch}, options = run_option)
            losses.append(ls_)
            
            if i > 0 and i % 20 == 0:
                writer.add_summary(sess.run(summary, feed_dict={X: X_batch}), e * X_data.shape[0] // batch_size + i)
                writer.flush()
            
        print('Epoch %d Loss %f' % (e, np.mean(losses)))
        losses = []

        gen_img = sess.run(g, feed_dict={X: [X_sample]})[0]
        gen_img = np.clip(gen_img, 0, 255)
        result = np.zeros((h_sample, w_sample * 2, 3))
        result[:, :w_sample, :] = X_sample / 255.
        result[:, w_sample:, :] = gen_img[:h_sample, :w_sample, :] / 255.
        # plt.axis('off')
        # plt.imshow(result)
        # plt.show()
        imsave(os.path.join(OUTPUT_DIR, 'sample_epoch_%d.jpg' % e), result)
        saver.save(sess, os.path.join(OUTPUT_DIR, 'trained_model'), global_step = e,  write_meta_graph = False)

    if(write_log):
        writer.close()

if(__name__ == '__main__'):
    main(sys.argv[1:])



