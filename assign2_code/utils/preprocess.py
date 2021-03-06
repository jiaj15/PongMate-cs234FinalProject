#!usr/bin/python3

import numpy as np
import tensorflow as tf
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2

def greyscale_tennis(state):
    state = np.reshape(state, [250, 160, 3]).astype(np.float32)
    # state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114
    state = state[:, :, 0] * 0.15 + state[:, :, 1] * 0.7 + state[:, :, 2] * 0.15
    state = state[45:215, 10:150]  # crop
    state = cv2.pyrDown(state)
    
    # state = state[::2,::2] # downsample by factor of 2
    state = state[:, :, np.newaxis]
    
    return state.astype(np.uint8)



def greyscale(state):
    """
    Preprocess state (210, 160, 3) image into
    a (80, 80, 1) image in grey scale
    """
    state = np.reshape(state, [210, 160, 3]).astype(np.float32)

    # grey scale
    state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114

    # karpathy
    state = state[35:195]  # crop
    state = state[::2,::2] # downsample by factor of 2

    state = state[:, :, np.newaxis]

    return state.astype(np.uint8)


def blackandwhite(state):
    """
    Preprocess state (210, 160, 3) image into
    a (80, 80, 1) image in grey scale
    """
    # erase background
    state[state==144] = 0
    state[state==109] = 0
    state[state!=0] = 1

    # karpathy
    state = state[35:195]  # crop
    state = state[::2,::2, 0] # downsample by factor of 2

    state = state[:, :, np.newaxis]

    return state.astype(np.uint8)