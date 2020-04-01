# -*- coding:utf-8 -*-
import numpy as np


# 数组排序
def array_extract():
    # 对于一个任意阵，写一个方法，根据确定的中心点以及大小来提取一个子阵（必要时给定填充值padding）
    Z = np.random.randint(0, 10, (10, 10))
    shape = (5, 5)
    fill = 0
    position = (1, 1)

    R = np.ones(shape, dtype=Z.dtype) * fill
    P = np.array(list(position)).astype(int)
    Rs = np.array(list(R.shape)).astype(int)
    Zs = np.array(list(Z.shape)).astype(int)

    R_start = np.zeros((len(shape),)).astype(int)
    R_stop = np.array(list(shape)).astype(int)
    Z_start = (P - Rs // 2)
    Z_stop = (P + Rs // 2) + Rs % 2

    R_start = (R_start - np.minimum(Z_start, 0)).tolist()
    Z_start = (np.maximum(Z_start, 0)).tolist()
    R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop - Zs, 0))).tolist()
    Z_stop = (np.minimum(Z_stop, Zs)).tolist()

    r = [slice(start, stop) for start, stop in zip(R_start, R_stop)]
    z = [slice(start, stop) for start, stop in zip(Z_start, Z_stop)]
    R[r] = Z[z]
    print(Z)
    print(R)
    
    # 从一个10*10的矩阵中提取出所有相邻的3*3块
    Z = np.random.randint(0, 5, (10, 10))
    n = 3
    i = 1 + (Z.shape[0] - 3)
    j = 1 + (Z.shape[1] - 3)
    C = np.lib.stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
    print(C)
    
    # 对于大小分别为（8，3）和（2，2）的数组A,B， 在A里找出包含有B里元素的行，在不考虑B的序列的情况下。
    A = np.random.randint(0, 5, (8, 3))
    B = np.random.randint(0, 5, (2, 2))

    C = (A[..., np.newaxis, np.newaxis] == B)
    rows = np.where(C.any((3, 1)).all(1))[0]
    print(rows)

    # 对于一个10*3的矩阵，提取出不单一值的每一行（比如[2,2,3]）
    Z = np.random.randint(0, 5, (10, 3))
    print(Z)
    # solution for arrays of all dtypes (including string arrays and record arrays)
    E = np.all(Z[:, 1:] == Z[:, :-1], axis=1)
    U = Z[~E]
    print(U)
    # soluiton for numerical arrays only, will work for any number of columns in Z
    U = Z[Z.max(axis=1) != Z.min(axis=1), :]
    print(U)
    
    # 一个二维数组，怎样提取独特的行
    Z = np.random.randint(0, 2, (6, 2))
    uZ = np.unique(Z, axis=0)
    print(uZ)
    
    # 给定一个整数n和一个2D数组X，从X中选择可以解释为从具有n度的多项式分布中得出的行，即仅包含整数且总和为n的行 
    X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                    [2.0, 0.0, 1.0, 1.0],
                    [1.5, 2.5, 1.0, 0.0]])
    n = 4
    M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
    M &= (X.sum(axis=-1) == n)
    print(X[M])
    
    # 考虑由两个向量（X，Y）描述的路径，如何使用等距样本对其进行采样
    phi = np.arange(0, 10 * np.pi, 0.1)
    a = 1
    x = a * phi * np.cos(phi)
    y = a * phi * np.sin(phi)

    dr = (np.diff(x) ** 2 + np.diff(y) ** 2) ** .5  # segment lengths
    r = np.zeros_like(x)
    r[1:] = np.cumsum(dr)  # integrate path
    r_int = np.linspace(0, r.max(), 200)  # regular spaced path
    x_int = np.interp(r_int, r, x)  # integrate path
    y_int = np.interp(r_int, r, y)
    
    
if __name__ == '__main__':
    array_extract()
