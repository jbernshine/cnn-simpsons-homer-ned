import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import os
import cv2
import random

TRAINDIR = "./training"
CATEGORIES = ["homer", "ned"]

IMG_SIZE = 70
training_data = []

X = []
y = []


def create_training_data():
    global X, y

    for category in CATEGORIES:
        path = os.path.join(TRAINDIR, category)  # path to dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    random.seed(5)  # same result every time
    random.shuffle(training_data)

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def train():
    global X, y

    X = X / 255.0

    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X, y, batch_size=32, epochs=4, validation_split=0.1)

    model.save("simpsons-homer-ned.model")


if __name__ == '__main__':
    create_training_data()
    train()
