import cv2
#import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt


image_width = 160
image_height = 90

batch_size = 60 # sampling at 60 times a second, so 1 batch equals 1 second

cap = cv2.VideoCapture('udp://127.0.0.1:9999',cv2.CAP_FFMPEG)

if not cap.isOpened():
  print('VideoCapture not opened')
  exit(-1)

while True:
    ret, src = cap.read()

    if not ret:
        print('frame empty')
        continue
    #src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    src = cv2.resize(src, (0,0), fx=.1, fy=.1)

    cv2.imshow('test', cv2.resize(src, (0,0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
