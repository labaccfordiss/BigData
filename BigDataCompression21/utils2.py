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


# root mean square error function
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))