# -*- coding: utf-8 -*-

import sys
sys.path.append('.')

import numpy as np
import caffe
import h5py
import cv2

class CNN(object):
    def __init__(self, net, model):
        self.net = net
        self.model = model
        try:
            self.cnn = caffe.Net(str(net), str(model), caffe.TEST)
        except:
            print 'Can not open %s, %s' % (net, model)
    
    def forward(self, data, layers=['ip3']):
        fake = np.zeros((len(data), 1, 1, 1))
        self.cnn.set_input_arrays(data.astype(np.float32), fake.astype(np.float32))
        self.cnn.forward()
        result = []
        for layer in layers:
            result.append(self.cnn.blobs[layer].data[0])
        return result

def load():
    with h5py.File('dataset/validation.h5', 'r') as hf:
        data = hf.get('data')
        data = np.array(data)
        landmark = hf.get('landmark')
        landmark = np.array(landmark)
    return (data, landmark)

def draw_landmark(x, y):
    img = x.reshape(96, 96)
    for i in range(15):
        cv2.circle(img, (int(y[i][0]), int(y[i][1])), 2, (0,255,0), -1)
    cv2.imshow("img", img)
    cv2.waitKey(0)
 
X, landmark = load()

cnn = CNN('prototxt/deploy.prototxt', 'model/_iter_3000.caffemodel')

print X.shape
for i in range(X.shape[0]):
    data = X[i].reshape(1,1,96,96)
    points = cnn.forward(data)[0].reshape(15, 2)
    points = points * 48. + 48
    draw_landmark(data, points)