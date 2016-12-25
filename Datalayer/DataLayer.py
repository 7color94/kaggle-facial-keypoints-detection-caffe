#!/usr/bin/env python
# -*- coding: utf-8 -*-

import caffe
import json
import h5py
import numpy as np

class KaggleDataLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        if len(bottom) > 0:
            raise Exception('cannot have bottoms for input layer')
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        param = json.loads(self.param_str)
        # print 'setup: ', param
        self.phase = param['phase']
        self.idx = 0
        self.batch_size = param['batch_size']
        self.file_name = param['file_name']
        self.batch_lodaer = BatchLoader(self.batch_size, self.file_name)
    
    def reshape(self, bottom, top):
        # print self.phase
        self.data, self.label = self.batch_lodaer.get_batch(self.idx)
        _n = self.data.shape[0]
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.label.shape)
    
    def forward(self, bottom, top):
        # Flip half of the images in this batch at random:
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        self.idx += 1
        # print 'forward: ', self.idx

    def backward(self, top, prapagate_down, bottom):
        # no back-propagate for data layers
        pass

class BatchLoader(object):
    def __init__(self, batch_size, file_name):
        self.batch_size = batch_size
        self.file_name = file_name
        self.flip_indices = [
            (0, 2), (1, 3),
            (4, 8), (5, 9), (6, 10), (7, 11),
            (12, 16), (13, 17), (14, 18), (15, 19),
            (22, 24), (23, 25),
        ]
        with h5py.File(self.file_name, 'r') as hf:
            data = hf.get('data')
            data = np.array(data)
            label = hf.get('landmark')
            label = np.array(label)
        self.data = data
        self.label = label
        self.samples_n = np.shape(self.label)[0]
        self._epoch = np.ceil(self.samples_n / self.batch_size)
        # print 'init BatchLoader done.'

    def get_batch(self, idx):
        indices = (idx % self._epoch) * self.batch_size
        # print 'get_batch: ', indices
        batch_data = self.data[indices:indices+self.batch_size]
        batch_label = self.label[indices:indices+self.batch_size]
        _n = np.shape(batch_data)[0]
        indices = np.random.choice(_n, _n / 2, replace=False)
        batch_data[indices] = batch_data[indices, :, :, ::-1]
        batch_label[indices, ::2] = batch_label[indices, ::2] * -1
        for a, b in self.flip_indices:
            batch_label[indices, a], batch_label[indices, b] = (batch_label[indices, b], batch_label[indices, a])
        return batch_data, batch_label