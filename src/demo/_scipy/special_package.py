# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from scipy.special import cbrt
from scipy.special import exp10
from scipy.special import exprel
from scipy.special import logsumexp
from scipy.special import lambertw
from scipy.special import comb
from scipy.special import perm
from scipy.special import gamma


def special_package():
    # 立方根函数
    res = cbrt([10, 9, 0.1254, 234])
    print(res)

    # 指数函数
    res = exp10([2, 4])
    print(res)

    # 相对误差指数函数
    res = exprel([-0.25, -0.1, 0, 0.1, 0.25])
    print(res)

    # 对数和指数函数
    a = np.arange(10)
    res = logsumexp(a)
    print(res)

    # 兰伯特函数
    w = lambertw(1)
    print(w)
    print(w * np.exp(w))

    # 排列和组合
    res = comb(10, 3, exact=False, repetition=True)
    print(res)

    # 排列
    res = perm(10, 3, exact=True)
    print(res)

    # 伽马函数
    res = gamma([0, 0.5, 1, 5])
    print(res)


if __name__ == '__main__':
    special_package()
