#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import h5py

FTRAIN = "dataset/training/training.csv"
TRAIN_O = 'dataset/train.h5'
VALIDATION_O = 'dataset/validation.h5'

def load():
    df = read_csv(FTRAIN)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    print df.count()
    df = df.dropna()
    
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)
    X = X.reshape(-1, 1, 96, 96)
    
    y = df[df.columns[:-1]].values
    y = (y - 48) / 48.0
    X, y = shuffle(X, y, random_state=42)
    y = y.astype(np.float32)

    return X, y

X, y = load()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

with h5py.File(TRAIN_O, 'w') as h5:
    h5['data'] = X_train
    h5['landmark'] = y_train

with h5py.File(VALIDATION_O, 'w') as h5:
    h5['data'] = X_test
    h5['landmark'] = y_test