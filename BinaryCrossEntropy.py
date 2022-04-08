# -*- Coding = UTF-8 -*-
# @Time: 2022/4/7 15:51
# @Author: Nico
# File: BinaryCrossEntropy.py
# @Software: PyCharm


import numpy as np
from numpy import log


def binary_cross_entropy(y, y_hat):
    loss = -y * log(y_hat) - (1 - y) * log(1 - y_hat)

    return loss


binary_cross_entropy(0, 0.01)
binary_cross_entropy(1, 0.99)
binary_cross_entropy(0, 0.3)
binary_cross_entropy(0, 0.8)

print(binary_cross_entropy(0, 0.01))
print(binary_cross_entropy(1, 0.99))
print(binary_cross_entropy(0, 0.3))
print(binary_cross_entropy(0, 0.8))

