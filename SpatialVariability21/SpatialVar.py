


import glob
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
print(tf.__version__)  # prints version of tensorflow installed
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import pickle
import scipy.stats as stats


#initialization
num_epochs = 80
N_FEATURES = 6
num_classes = 6
frame_size = 100
hop_size = 100
input_shape = (6, 100, 1)

file = "./"
training_indices='training_indices/'
testing_indices= '/testing_indices/'

def swpad_run(file):
    # Load and process the Dataset
    data = file
    data.reset_index()
    print(data.info())
    print(data.head())
    sname = data['sensor_name'][0]
    print("Analysis of channel = ", sname)
    activities = data['activity_name'].value_counts().index
    df = data.drop(['index',"index.1", 'attr_time', 'attr_time_y'], axis=1).copy()
    print("df.info = ", df.info())
    df['activity_name'].value_counts()
    Climbingup = df[df['activity_name'] == 'climbingup'].copy()
    Climbingdown = df[df['activity_name'] == 'climbingdown'].copy()
    Running = df[df['activity_name'] == 'running'].copy()
    Jumping = df[df['activity_name'] == 'jumping'].copy()
    Walking = df[df['activity_name'] == 'walking'].copy()
    Lying = df[df['activity_name'] == 'lying'].copy()

    balanced_data = pd.DataFrame()
    balanced_data = balanced_data.append([Climbingup, Climbingdown, Running, Jumping, Walking, Lying])
    balanced_data['activity_name'].value_counts()
    label = LabelEncoder()
    balanced_data['activity'] = label.fit_transform(balanced_data['activity_name'])
    lcc=label.classes_
    X = balanced_data[['attr_ax', 'attr_ay', 'attr_az', 'attr_gx', 'attr_gy', 'attr_gz']]
    y = balanced_data['activity']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaled_X = pd.DataFrame(data=X, columns=['attr_ax', 'attr_ay', 'attr_az', 'attr_gx', 'attr_gy', 'attr_gz'])
    scaled_X['activity'] = y.values
    print(y.values)
    print(scaled_X.head())

    def get_frames(df, frame_size, hop_size):

        frames = []
        labels = []
        for i in range(0, len(df) - frame_size, hop_size):
            t = df['attr_ax'].values[i: i + frame_size]
            u = df['attr_ay'].values[i: i + frame_size]
            v = df['attr_az'].values[i: i + frame_size]
            w = df['attr_gx'].values[i: i + frame_size]
            x = df['attr_gy'].values[i: i + frame_size]
            z = df['attr_gz'].values[i: i + frame_size]
            # Retrieve the most often used label in this segment
            label = stats.mode(df['activity'][i: i + frame_size])[0][0]
            frames.append([t, u, v, w, x, z])
            labels.append(label)

        # Bring the segments into a better shape
        frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
        labels = np.asarray(labels)
        print(labels)
        return frames, labels


    X, y = get_frames(scaled_X, frame_size, hop_size)

    #train test split

    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(X_train.shape, X_test.shape)
    print(X_test.shape, y_test.shape)
    print(X_train[0].shape, X_test[0].shape)
    total = len(X)
    testshape = int(total * 0.2)
    trainshape = int(total * 0.8)
    X_train = X_train.reshape(trainshape, 6, 100, 1)
    X_test = X_test.reshape(testshape, 6, 100, 1)
    print(X_train[0].shape, X_test[0].shape)


    ###
    model = Sequential()

    model.add(Conv2D(16, (1, 2), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((1, 2)))
    model.add(Dropout(0.15))
    model.add(Conv2D(32, (1, 2), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((1, 2)))
    model.add(Dropout(0.15))
    model.add(Conv2D(32, (1, 2), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((1, 2)))
    model.add(Dropout(0.15))
    model.add(Conv2D(64, (1, 2), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((1, 2)))
    model.add(Dropout(0.15))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.20))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=num_epochs, verbose=2)
    y_pred = model.predict_classes(X_test)

    history1 = history.history



files = sorted(glob.glob(file+"*.csv"))
df = pd.DataFrame()
for f in files:
    csvc = pd.read_csv(f)
    swpad_run(csvc)



