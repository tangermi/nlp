# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import uniform


def stats_demo():
    # 正态连续随机变量
    cdfarr = norm.cdf(np.array([1, -1., 0, 1, 3, 4, -2, 6]))
    print(cdfarr)

    # 查找分布的中位数
    ppfvar = norm.ppf(0.5)
    print(ppfvar)

    # 生成随机变量序列
    rvsvar = norm.rvs(size=5)
    print(rvsvar)

    # 使用统一函数生成均匀分布
    cvar = uniform.cdf([0, 1, 2, 3, 4, 5], loc=1, scale=4)
    print(cvar)

    # 二项分布
    cvar = uniform.cdf([0, 1, 2, 3, 4, 5], loc=1, scale=4)
    print(cvar)

    # 描述性统计
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(x.max(), x.min(), x.mean(), x.var())

    # 计算一组分数平均值的T检验
    rvs = stats.norm.rvs(loc=5, scale=10, size=(50, 2))
    sta = stats.ttest_1samp(rvs, 5.0)
    print(sta)

    # 比较两个样本
    rvs1 = stats.norm.rvs(loc=5, scale=10, size=500)
    rvs2 = stats.norm.rvs(loc=5, scale=10, size=500)
    print(stats.ttest_ind(rvs1, rvs2))


if __name__ == '__main__':
    stats_demo()
