
import numpy as np
import pandas as pd
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras import backend as K
from math import sqrt
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from enum import Enum
from numpy import array, loadtxt
import tensorflow as tf
import os
import copy

from utils1 import *
import matplotlib.pyplot as plt


###############################################
NumL_arr = [2,4]
InWidth_arr = [128]
FMDelta_arr = [2,4,6]
ConvSize_arr = [5]

MPSize = 2
epochs = 100
batch_size = 8
use_bias = True
relu_or_tanh = 1

verbose = 0
fileDir_read = './'
sub_arr = [1,2,3,4,5]


Train_pct = 0.90

###############################################
for id, sub_number in zip(range(len(sub_arr)), sub_arr):
        fileread_txt = f'{fileDir_read}Sub{sub_number}_data.txt'
        wave_data = loadtxt(fileread_txt, delimiter=',', skiprows=1)
        print("Loaded Data_all shapes:", wave_data.shape)

        for i_NumL, NumL in zip(range(len(NumL_arr)), NumL_arr):
            for i_InWidth, InWidth in zip(range(len(InWidth_arr)), InWidth_arr):
                # prepare train/test data
                num_ins = int(np.floor(len(wave_data)/InWidth))
                new_data    = wave_data[0:num_ins*InWidth].reshape(num_ins, InWidth)
                new_data = new_data.reshape(new_data.shape[0], InWidth, 1)

                train_size  = int(num_ins * Train_pct)
                test_size   = num_ins - train_size
                x_train = new_data[0:train_size]
                x_test = new_data[train_size:]
                print('\nTraining data shape : ', x_train.shape)
                print('Testing data shape : ', x_test.shape)

                x_tr_max = np.max(x_train)
                x_tr_min = np.min(x_train)
                x_train = (x_train - np.min(x_train)) / (x_tr_max - x_tr_min)
                x_test =  (x_test - np.min(x_test))   /  (x_tr_max - x_tr_min)

                for i_FMDelta, FMDelta in zip(range(len(FMDelta_arr)), FMDelta_arr):
                    for i_ConvSize, ConvSize in zip(range(len(ConvSize_arr)), ConvSize_arr):
                        in_shape = x_train.shape[1:]
                        # encoding layers
                        use_bias = use_bias
                        input_window = Input(shape=in_shape, name='e_input')
                        x = input_window

                        x = LSTM(units=1, return_sequences=True, activation= 'tanh', name='e_lstm')(x)

                        for l in range(NumL-1,0,-1):
                            x = Conv1D(FMDelta, ConvSize, activation= 'tanh', padding="same", use_bias=use_bias, name=f'e_COV_{l}')(x)
                            x = MaxPooling1D(MPSize, padding="same", name=f'e_MAX_{l}')(x)

                        x = Conv1D(1, ConvSize, activation= 'tanh', padding="same", use_bias=use_bias,  name=f'e_COV_o')(x)
                        encoded = MaxPooling1D(MPSize, padding="same",name=f'e_MAX_o')(x)

                        # decoding layers
                        x = UpSampling1D(MPSize,name=f'd_UPSM_i')(encoded)
                        x = Conv1D(FMDelta, ConvSize, activation='tanh' if relu_or_tanh==0 else 'tanh', padding="same", use_bias=use_bias,  name=f'd_COVT_i')(x)

                        for l in range(1,NumL,1):
                            x = UpSampling1D(MPSize, name=f'd_UPSM_{l}')(x)
                            x = Conv1D(FMDelta, ConvSize, activation='relu' if relu_or_tanh==0 else 'tanh', padding='same', use_bias=use_bias,  name=f'd_COVT_{l}')(x)

                        decoded       = LSTM(units=1,             return_sequences=True)(x)


                        encoder     = Model(input_window, encoded)
                        autoencoder = Model(input_window, decoded)

                        autoencoder.compile(optimizer='rmsprop', loss=tf.keras.losses.MeanSquaredError())


                        ### train & test
                        print('Training...')
                        history2        = autoencoder.fit(x_train, x_train, epochs=epochs*1, batch_size=batch_size, verbose=verbose, shuffle=True, )#verbose:0/1/2-no/all/simple
                        x_pred          = autoencoder.predict(x_test)


                        ## plot of train curve
                        loss_train = history2.history['loss']
                        fig = plt.figure()
                        fig.set_size_inches(5, 4)

                        ax = fig.add_subplot(1, 1, 1)
                        ax.plot(loss_train, 'g', label='Training loss')
                        plt.title('Training loss')
                        plt.xlabel('Epochs')
                        plt.ylabel('Loss')

                        ## plot of selected decoding results
                        fig = plt.figure()
                        fig.set_size_inches(10, 7)

                        for i in range(0,16):
                            ax = fig.add_subplot(4, 4, i+1)

                            # # debug
                            x_test_this = x_test[i+0]
                            x_pred_this = x_pred[i+0]

                            ax.plot(x_test_this, 'g', label='x_test')
                            ax.plot(x_pred_this, 'b', label='x_pred')
                            plt.legend(loc="lower right")
