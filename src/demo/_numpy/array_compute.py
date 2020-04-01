# -*- coding:utf-8 -*-
import numpy as np


# 数组计算
def array_compute():
    # 怎样对数组求和，速度要快于np.sum
    Z = np.arange(10)
    print(np.add.reduce(Z))

    # 判断2个随机数组A和B是否相等
    A = np.random.randint(0, 2, 5)
    B = np.random.randint(0, 2, 5)
    equal = np.allclose(A, B)  # Assuming identical shape of the arrays and a tolerance for the comparison of values
    print(equal)
    equal = np.array_equal(A,
                           B)  # Checking both the shape and the element values, no tolerance (values have to be exactly equal)
    print(equal)

    # 使用一个3*2的矩阵乘以一个5*3的矩阵（矩阵积）
    Z = np.dot(np.ones((5, 3)), np.ones((3, 2)))
    print(Z)
    # 第二种方法， 在python3.5以上版本有效
    Z = np.ones((5, 3)) @ np.ones((3, 2))
    print(Z)

    # 一个大小为100*2的随机的向量来表达坐标系，找出点与点之间的距离
    Z = np.random.random((10, 2))
    X, Y = np.atleast_2d(Z[:, 0], Z[:, 1])
    D = np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2)
    print(D)

    # Much faster with scipy
    import scipy
    import scipy.spatial

    Z = np.random.random((10, 2))
    D = scipy.spatial.distance.cdist(Z, Z)
    print(D)

    # 矩阵的减去每行的平均值
    X = np.random.rand(5, 10)
    print(X)
    Y = X - X.mean(axis=1, keepdims=True)  # 当前版本numpy
    # Y = X - X.mean(axis=1).reshape(-1, 1)   # 老版本numpy
    print(Y)

    # 检测一个2维数组 是否包含null列
    Z = np.random.randint(0, 3, (3, 10))
    print((~Z.any(axis=0)).any())

    # 从一个数组里，找出距离给出数值最近的值
    Z = np.random.uniform(0, 1, 10)
    z = 0.5
    m = Z.flat[np.abs(Z - z).argmin()]
    print(m)

    # 2个数组的大小分别为(1,3)和(3,1)，使用iterator计算他们的和
    A = np.arange(3).reshape(3, 1)
    B = np.arange(3).reshape(1, 3)
    it = np.nditer([A, B, None])
    for x, y, z in it: z[...] = x + y
    print(it.operands[2])

    # 对于一个向量，根据另一个向量以bincount的方式来给他添加1
    Z = np.ones(10)
    I = np.random.randint(0, len(Z), 20)
    Z += np.bincount(I, minlength=len(Z))
    print(Z)
    # np.add.at(Z, I, 1)   # 方法2
    # print(Z)

    # 根据一个index listI积累向量X的元素到数组F上
    X = [1, 2, 3, 4, 5, 6]
    I = [1, 3, 9, 3, 4, 1]
    F = np.bincount(I, X)
    print(F)

    # 一个(w,h,3)图片(dtype=ubyte), 计算独特颜色的数量
    w, h = 16, 16
    I = np.random.randint(0, 2, (h, w, 3)).astype(np.ubyte)
    print(I)
    # Note that we should compute 256*256 first.
    # Otherwise numpy will only promote F.dtype to 'uint16' and overfolw will occur
    F = I[..., 0] * (256 * 256) + I[..., 1] * 256 + I[..., 2]
    n = len(np.unique(F))
    print(n)

    # 一个4维的数组，获取最后2个维度的和
    A = np.random.randint(0, 10, (3, 4, 3, 4))
    # solution by passing a tuple of axes (introduced in numpy 1.7.0)
    sum = A.sum(axis=(-2, -1))
    print(sum)
    # solution by flattening the last two dimensions into one
    # (useful for functions that don't accept tuples for axis argument)
    sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
    print(sum)

    # 对于一个一维的向量D，使用一个同大小的向量S作为index来计算它的子集平均值
    D = np.random.uniform(0, 1, 100)
    S = np.random.randint(0, 10, 100)
    D_sums = np.bincount(S, weights=D)
    D_counts = np.bincount(S)
    D_means = D_sums / D_counts
    print(D_means)
    # Pandas solution as a reference due to more intuitive code
    import pandas as pd
    print(pd.Series(D).groupby(S).mean())

    # 获取点积的对角线
    A = np.random.uniform(0, 1, (5, 5))
    B = np.random.uniform(0, 1, (5, 5))
    # Slow version  
    np.diag(np.dot(A, B))
    # Fast version
    np.sum(A * B.T, axis=1)
    # Faster version
    np.einsum("ij,ji->i", A, B)

    # 一个维度为(5,5,3)的数组，让他与一个维度为(5,5)的数组相乘
    A = np.ones((5, 5, 3))
    B = 2 * np.ones((5, 5))
    print('-' * 20)
    print(B[:, :, None])
    print('-' * 20)
    print(A * B[:, :, None])

    # 对于2组点P0,P1，来表达在2维平面上的线，计算p到每一条线的距离(P0[i],P1[i])
    def distance(P0, P1, p):
        T = P1 - P0
        L = (T ** 2).sum(axis=1)
        U = -((P0[:, 0] - p[..., 0]) * T[:, 0] + (P0[:, 1] - p[..., 1]) * T[:, 1]) / L
        U = U.reshape(len(U), 1)
        D = P0 + U * T - p
        return np.sqrt((D ** 2).sum(axis=1))

    P0 = np.random.uniform(-10, 10, (10, 2))
    P1 = np.random.uniform(-10, 10, (10, 2))
    p = np.random.uniform(-10, 10, (1, 2))
    print(distance(P0, P1, p))

    # 对于2组点P0,P1，来表达在2维平面上的线，还有一组点P，计算每一个点P[j]到每条线(P0[i],P1[i])的距离
    P0 = np.random.uniform(-10, 10, (10, 2))
    P1 = np.random.uniform(-10, 10, (10, 2))
    p = np.random.uniform(-10, 10, (10, 2))
    print(np.array([distance(P0, P1, p_i) for p_i in p]))  # 使用上一个问题的distance方法

    # 使用滑动窗口计算数组的平均值
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    Z = np.arange(20)
    print(moving_average(Z, n=3))

    # 求一个布尔值的否值，改变float数的正负
    Z = np.random.randint(0, 2, 100)
    np.logical_not(Z, out=Z)
    Z = np.random.uniform(-1.0, 1.0, 100)
    np.negative(Z, out=Z)
    
    # 对于一组p矩阵n*n和一组p向量n*1，计算p矩阵乘积和（结果大小为n,1）
    p, n = 10, 20
    M = np.ones((p, n, n))
    V = np.ones((p, n, 1))
    S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
    print(S)
    # It works, because:
    # M is (p,n,n)
    # V is (p,n,1)
    # Thus, summing over the paired axes 0 and 0 (of M and V independently),
    # and 2 and 1, to remain with a (n,1) vector.

    # 对于一个16*16数组，计算出4*4的块的和
    Z = np.ones((16, 16))
    k = 4
    S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                        np.arange(0, Z.shape[1], k), axis=1)
    print(S)
    
    # 计算矩阵秩
    Z = np.random.uniform(0, 1, (10, 10))
    U, S, V = np.linalg.svd(Z)  # Singular Value Decomposition
    rank = np.sum(S > 1e-10)
    print(rank)

    # 对于一个大向量Z，使用3种不同的方法计算它的立方
    x = np.random.rand(int(5e7))
    print(np.power(x, 3))
    print(x * x * x)
    np.einsum('i,i,i->i', x, x, x)
    
    # 为一维数组X的平均值计算自举的95％置信区间（即，用替换N次对数组的元素进行重新采样，计算每个样本的平均值，然后计算平均值的百分位数
    X = np.random.randn(100)  # random 1D array
    N = 1000  # number of bootstrap samples
    idx = np.random.randint(0, X.size, (N, X.size))
    means = X[idx].mean(axis=1)
    confint = np.percentile(means, [2.5, 97.5])
    print(confint)

# 对于随机数量的向量，构建笛卡尔乘积（每个物品的组合）
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

def do_cartesian():
    print(cartesian(([1, 2, 3], [4, 5], [6, 7])))

# 用Numpy来模拟Game of Life
def iterate(Z):
    # Count neighbours
    N = (Z[0:-2, 0:-2] + Z[0:-2, 1:-1] + Z[0:-2, 2:] +
            Z[1:-1, 0:-2] + Z[1:-1, 2:] +
            Z[2:, 0:-2] + Z[2:, 1:-1] + Z[2:, 2:])

    # Apply rules
    birth = (N == 3) & (Z[1:-1, 1:-1] == 0)
    survive = ((N == 2) | (N == 3)) & (Z[1:-1, 1:-1] == 1)
    Z[...] = 0
    Z[1:-1, 1:-1][birth | survive] = 1
    return Z

def do_iterate():
    Z = np.random.randint(0, 2, (50, 50))
    for i in range(100):
        Z = iterate(Z)
    print(Z)

def run():
    for task in tasks:
        eval(task)()

tasks = ['array_compute','do_cartesian' , 'do_iterate']


if __name__ == '__main__':
    run()

