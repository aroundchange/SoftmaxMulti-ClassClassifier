# -*- Coding = UTF-8 -*-
# @Time: 2022/4/7 15:57
# @Author: Nico
# File: CategoricalCrossEntropy.py
# @Software: PyCharm


import numpy as np
from numpy import log


def categorical_cross_entropy(y, y_hat):
    num_elements = len(y)
    loss = 0
    for i in range(num_elements):
        loss += - y[i] * log(y_hat[i])

    return loss


y = [0, 0, 1]
y_hat = [0.1, 0.1, 0.8]
loss = categorical_cross_entropy(y, y_hat)
print(loss)

y = [0, 0, 1]
y_hat = [0.1, 0.3, 0.6]
loss = categorical_cross_entropy(y, y_hat)
print(loss)

y = [0, 0, 1]
y_hat = [0.4, 0.5, 0.1]
loss = categorical_cross_entropy(y, y_hat)
print(loss)

