from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data

#Based off code found here:
    #http://machinelearninguru.com/deep_learning/tensorflow/neural_networks/autoencoder/autoencoder.html
#It's a really good tutorial and gave me a great jump start


batch_size = 500  # Number of samples in each batch
epoch_num = 5     # Number of epochs to train the network
lr = 0.001        # Learning rate


def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    # Args:
    #   imgs: a numpy array of size [batch_size, 28 X 28].
    # Returns:
    #   a numpy array of size [batch_size, 32, 32].
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs


def encoder(inputs):
    # encoder
    # 32 x 32 x 1   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  8 x 8 x 16
    # 8 x 8 x 16    ->  2 x 2 x 8
    net = lays.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [5, 5], stride=4, padding='SAME')
    return net

def decoder(inputs):
    # decoder
    # 2 x 2 x 8    ->  8 x 8 x 16
    # 8 x 8 x 16   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  32 x 32 x 1
    net = lays.conv2d_transpose(inputs, 16, [5, 5], stride=4, padding='SAME')
    net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    return net

def autoencoder(inputs):
    net = encoder(inputs)
    net = decoder(net)
    return net

# read MNIST dataset
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# calculate the number of batches per epoch
#batch_per_ep = mnist.train.num_examples // batch_size

ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 1))  # input to the network (MNIST images)
ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network

# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.div(tf.abs(ae_inputs - ae_outputs), 20))  # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# initialize the network
init = tf.global_variables_initializer()

cap = cv2.VideoCapture('udp://127.0.0.1:9999', cv2.CAP_FFMPEG)

if not cap.isOpened():
  print('VideoCapture not opened')
  exit(-1)

batch = []
curr_epoch = 0

for i in range(batch_size * epoch_num):
    ret, src = cap.read()
    if not ret:
        print('frame empty')
        i-=1
        continue
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    src = cv2.resize(src, (0,0), fx=.1, fy=.1)

    display = cv2.resize(src, (0,0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('data', display)

    batch.append(src[:, :, None])
    print(i)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
        for batch_n in range(500):  # batches loop
            batch_img = []
            batch_label = []
            for i in range(batch_size * (curr_epoch-1), batch_size * curr_epoch):
                batch_img.append(batch[i])
            batch_img = np.asarray(batch_img)
            batch_img = batch_img.reshape((-1, 32, 32, 1))               # reshape each sample to an (28, 28) image
            #batch_img = resize_batch(batch_img)                          # reshape the images to (32, 32)
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img})
            print('Epoch: {} - cost= {:.5f} batch: {}'.format((ep + 1), c, batch_n))
        curr_epoch += 1
    # test the trained network
    #batch_img, batch_label = mnist.test.next_batch(50)
    batch_img = batch
    batch_img = np.asarray(batch_img)
    #batch_img = resize_batch(batch_img)
    recon_img = sess.run([ae_outputs], feed_dict={ae_inputs: batch_img})[0]
    print(recon_img[0].shape)
    # plot the reconstructed images and their ground truths (inputs)

    while(True):
        for i in range(batch_size * epoch_num):
            cv2.imshow('orig ', cv2.resize(batch_img[i], (0,0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST))
            cv2.imshow('recon', cv2.resize(recon_img[i], (0,0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST))
            cv2.waitKey(20)

cap.release()
cv2.destroyAllWindows()
