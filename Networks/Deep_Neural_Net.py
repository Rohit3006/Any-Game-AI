from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import cv2
import os
import pandas as pd
from keras.utils import to_categorical


def replace(df, column, x):
    unique = list(set(df[column].values))
    return unique.index(x)


def create_deep_neural_net(output_neurons=1, layers=1, size_x=20, size_y=20):
    model = Sequential()

    model.add(Dense(size_x*size_y, activation='relu'))
    for i in range(layers):
        model.add(Dense(500-i*10, activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_neurons, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train(model, epochs=10, verbose=2, size_x=20, size_y=20, train_size=0.7, roi=[]):
    file_dir = "Frames/"
    x = []

    np.random.seed(1)

    files = sorted(os.listdir(file_dir))

    for file in files:
        file_name = os.path.join(file_dir, file)
        array = cv2.imread(file_name)
        array = array[roi[0]:roi[1], roi[2]:roi[3]]
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        array = cv2.Canny(array, 100, 200)
        cv2.imwrite("img.png", array)

        array = cv2.resize(array, (size_x, size_y))
        x.append(preprocessing.scale(array))

    df = pd.read_csv("inputs.csv")
    y_set = set(np.array(df["Input"]))
    y = np.array(df["Input"].apply(lambda x: replace(df, "Input", x)))

    x = np.array(x)
    y = to_categorical(y, len(set(y)))

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)
    x_train = x_train.reshape(-1,size_x*size_y)
    x_test = x_test.reshape(-1, size_x*size_y)

    model.fit(x_train, y_train, epochs=epochs, verbose=verbose,
              validation_data=(x_test, y_test))

    model.save("model.h5")
    return list(y_set), model