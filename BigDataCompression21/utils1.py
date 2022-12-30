import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose  #add tensorflow at beginning
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras import backend as K
#import wfdb
import os
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from scipy.signal import lfilter, freqz
from scipy.signal import butter, filtfilt

import matplotlib
#matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt

import statistics
from scipy import stats




def butter_lowpass(lowcutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = lowcutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, lowcutoff, fs, order):
    b, a = butter_lowpass(lowcutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(highcutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = highcutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, highcutoff, fs, order=5):
    b, a = butter_highpass(highcutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

