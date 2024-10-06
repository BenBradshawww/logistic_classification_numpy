import numpy as np


def accuracy(y_true, y_pred):

    n = len(y_true)

    return 1 - (np.sum(np.abs(y_true - y_pred)) / n)


def precision(y_true, y_pred):

    n = len(y_true)

    for i in range(n):
        if y_true[i] == 1 and y_pred[i] == 1:
            temp += 1
            num += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            num += 1

    return temp / num
