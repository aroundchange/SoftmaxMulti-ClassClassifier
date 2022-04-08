# -*- Coding = UTF-8 -*-
# @Time: 2022/4/7 20:10
# @Author: Nico
# File: SoftmaxClassifierMINSTDataset.py
# @Software: PyCharm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical  # One-Hot Encoding

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# plt.imshow(X_train[0])
# plt.show()
# print(y_train[0])

n_train = X_train.shape[0]  # 60000
n_test = X_test.shape[0]  # 10000

flatten_size = 28 * 28

# X_train(60000 * 28 * 28)
X_train = X_train.reshape((n_train, flatten_size))
# 0 ~ 255   -->   0 ~ 1
X_train = X_train / 255
# 0 ~ 9   -->   One-Hot Encoding
# 3   -->   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train)

X_test = X_test.reshape((n_test, flatten_size))
X_test = X_test / 255
y_test = to_categorical(y_test)

X_train.shape
# print(X_train.shape)
y_train.shape
# print(y_train.shape)
y_train[0]
# print(y_train[0])


def softmax(x):
    exp_x = np.exp(x)
    sum_e = np.sum(exp_x, axis=1)
    for i in range(x.shape[0]):
        exp_x[1, :] = exp_x[i, :] / sum_e[i]
    return exp_x


W = np.zeros((784, 10))
b = np.zeros((1, 10))

N = 100
lr = 0.00001

for i in range(N):
    det_w = np.zeros((784, 10))
    det_b = np.zeros((1, 10))

    logits = np.dot(X_train, W) + b
    y_hat = softmax(logits)

    det_w = np.dot(X_train.T, (y_hat - y_train))
    det_b = np.sum((y_hat - y_train), axis=0)

    W = W - lr * det_w
    b = b - lr * det_b

logits_train = np.dot(X_train, W) + b
y_train_hat = softmax(logits_train)

y_hat = np.argmax(y_train_hat, axis=1)
y = np.argmax(y_train, axis=1)

count = 0
for i in range(len(y_hat)):
    if y[i] == y_hat[i]:
        count += 1

print('Accuracy On Training Set Is {}%'.format(round(count / n_train, 2) * 100))

logits_test = np.dot(X_test, W) + b
y_test_hat = softmax(logits_test)

y_hat = np.argmax(y_test_hat, axis=1)
y = np.argmax(y_test, axis=1)

count = 0
for i in range(len(y_hat)):
    if y[i] == y_hat[i]:
        count += 1

print('Accuracy On Testing Set Is {}%'.format(round(count / n_test, 2) * 100))

