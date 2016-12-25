#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')

import os

def w(c):
    if c != 0:
        print '\n'
        print ':-('
        print '\n'
        sys.exit()

def runCommand(cmd):
    w(os.system(cmd))


def train():
   cmd = "caffe/build/tools/caffe train --solver prototxt/solver.prototxt -gpu 3 2>&1 | tee log/train.log"
   runCommand(cmd)

if __name__ == '__main__':
    train()
