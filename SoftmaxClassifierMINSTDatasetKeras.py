# -*- Coding = UTF-8 -*-
# @Time: 2022/4/7 23:21
# @Author: Nico
# File: SoftmaxClassifierMINSTDatasetKeras.py
# @Software: PyCharm


import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import RMSprop

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(units=10, activation='softmax'))
# model.summary()

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer=RMSprop())
model.fit(X_train, y_train, epochs=10, batch_size=64)

loss, accuracy = model.evaluate(X_test, y_test)
print(accuracy)

