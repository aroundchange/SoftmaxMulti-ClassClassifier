# -*- Coding = UTF-8 -*-
# @Time: 2022/4/7 13:57
# @Author: Nico
# File: Softmax.py
# @Software: PyCharm


import numpy as np


def softmax(array):
    array = np.array(array)
    exp_array = np.exp(array)
    exp_sum = np.sum(exp_array)

    return exp_array / exp_sum


softmax([1, 3, 5])
softmax([1, 2, 3, 4, 5])

print(softmax([1, 3, 5]))
print(softmax([1, 2, 3, 4, 5]))

