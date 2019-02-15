from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import cv2
import os
import pickle
import pandas as pd
import glob
from keras.utils import to_categorical

def replace(df, column, x):
    unique = list(set(df[column].values))
    return unique.index(x)


def create_conv_net(output_neurons=1, layers=1, epochs=10, verbose=2, size_x=20, size_y=20, train_size=0.7):
    file_dir = "Frames/"
    x = []

    np.random.seed(1)

    files = sorted(os.listdir(file_dir))

    for file in files:
        file_name = os.path.join(file_dir, file)
        array = cv2.imread(file_name)
        array = cv2.resize(array, (size_y, size_x))
        x.append(array)

    x = np.array(x) / 255
    print(x[0].shape)

    df = pd.read_csv("inputs.csv")
    y_set = set(np.array(df["Input"]))
    y = np.array(df["Input"].apply(lambda x: replace(df, "Input", x)))
    y = to_categorical(y, len(set(y)))

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)

    model = Sequential()

    model.add(Conv2D(32, (3,3)))
    model.add(MaxPool2D((2,2)))
    for i in range(layers):
        model.add(Conv2D(16, (2, 2)))
        model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(output_neurons, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, verbose=verbose, validation_data=(x_test, y_test))

    model.save("model.model")

    return list(y_set), model
