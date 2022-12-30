
# the imports that will be used
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras import backend as K
import os
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from scipy.signal import lfilter, freqz
from scipy.signal import butter, filtfilt
import statistics
from scipy import stats
from numpy import array, loadtxt
from enum import Enum
import matplotlib.pyplot as plt


##Plot
wave_data = ''
idx_choose = ''
data_choose = wave_data[idx_choose[0]:idx_choose[1]]

fig = plt.figure()
ax = fig.add_subplot(111)
fig.set_size_inches(6, 5)
ax.plot(data_choose, '0.5', linewidth=2, color='b')

plt.show()


##plot
idx_choose = [128*10,128*10+128*3]
data_choose = wave_data[idx_choose[0]:idx_choose[1]]

fig = plt.figure()
ax = fig.add_subplot(111)
fig.set_size_inches(5, 4)
ax.plot(data_choose, '0.5', linewidth=1, color='b')

ax.set_xlabel('Sample index');
ax.set_ylabel('Amplitude')

plt.show()