# -*-coding:utf8-*-
import numpy as np


# 归一化方法
class Normalization:
    # Min-Max Feature scaling
    @staticmethod
    def min_max_scale(x):
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x - min_val) / (max_val - min_val)
        return x

    # standard score
    @staticmethod
    def standard_score(x, mean, std):
        res = (x - mean) / std
        return res

    @staticmethod
    def mean_normalise(x):
        mean_val = np.mean(x)
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x - mean_val) / (max_val - min_val)
        return x

    @staticmethod
    def unit_length(x):
        x = x / np.abs(x)
        return x
