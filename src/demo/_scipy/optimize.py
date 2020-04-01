# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import root


def optimize():
    # Nelder–Mead单纯形算法
    def rosen(x):
        x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
        res = minimize(rosen, x0, method='nelder-mead')
        print(res.x)

    # 最小二乘, 求解一个带有变量边界的非线性最小二乘问题
    def fun_rosenbrock(x):
        return np.array([10 * (x[1] - x[0] ** 2), (1 - x[0])])

    input = np.array([2, 2])
    res = least_squares(fun_rosenbrock, input)
    print(res)

    # 求根
    def func(x):
        return x * 2 + 2 * np.cos(x)

    sol = root(func, 0.3)
    print(sol)


if __name__ == '__main__':
    optimize()
