import cv2
import tensorflow as tf
import tensorflow.contrib.layers as lays
import numpy as np
import matplotlib.pyplot as plt

image_size = 160

batch_size = 60 # sampling at 60 times a second, so 1 batch equals 1 second
lr = 0.001      # learning rate
epoch_num = 1   # I'm always using unique data

def encoder(inputs):
    net = lays.conv2d(inputs, 160, [5, 5], stride=2, padding ='SAME');
    net = lays.conv2d(net, 80, [5, 5], stride=2, padding ='SAME');
    net = lays.conv2d(net, 40, [5, 5], stride=4, padding ='SAME');
    return net

def decoder(inputs):
    net = lays.conv2d_transpose(inputs, 80, [5, 5], stride=4, padding='SAME')
    net = lays.conv2d_transpose(net, 160, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 3, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    return net

def autoencoder(inputs):
    net = encoder(inputs)
    net = decoder(net);
    return net

def main_loop():
    cap = cv2.VideoCapture('udp://127.0.0.1:9999',cv2.CAP_FFMPEG)

    if not cap.isOpened():
      print('VideoCapture not opened')
      exit(-1)

    while True:
        batch_img = []
        for i in range(0, batch_size):
            ret, src = cap.read()

            if not ret:
                print('frame empty')
                i-=1
                continue

            #src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            np_src = np.asarray(src)
            batch_img.append(np_src)
            cv2.imshow('in', np_src)

            np_dst = sess.run([ae_outputs], feed_dict={ae_inputs: np.asarray(np_src[None, :])})[0][0]
            cv2.imshow('out', np_dst)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        batch = np.asarray(batch_img)
        _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch})

#tensorflow stuff
ae_inputs = tf.placeholder(tf.float32, (None, image_size, image_size, 3))
ae_outputs = autoencoder(ae_inputs)

loss = tf.reduce_mean(tf.square(ae_inputs - ae_outputs))
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#config.gpu_options.allow_growth = True

init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(sess.run(init))

main_loop()
