# -*- coding:utf-8 -*-
import numpy as np


# 数组的处理
def array_process():
    # 对一个一维数组，把值为3-8之间的元素变为负数
    Z = np.arange(11)
    Z[(3 < Z) & (Z <= 8)] *= -1
    print(Z)

    # 怎样取远离0的近似值
    Z = np.random.uniform(-10, +10, 10)
    print(Z)
    print(np.copysign(np.ceil(np.abs(Z)), Z))

    # 从随机数组中提取整数部分，使用5种不同的方法
    Z = np.random.uniform(0, 10, 10)
    print(Z - Z % 1)
    print(np.floor(Z))
    print(np.ceil(Z) - 1)
    print(Z.astype(int))
    print(np.trunc(Z))

    # 对一个5*5的随机数组进行归一化处理
    Z = np.random.random((5, 5))
    Z = (Z - np.mean(Z)) / (np.std(Z))
    print(Z)

    # 一个大小为10*2的随机向量来表达笛卡尔坐标系，转化为极坐标系
    Z = np.random.random((10, 2))
    X, Y = Z[:, 0], Z[:, 1]
    R = np.sqrt(X ** 2 + Y ** 2)
    T = np.arctan2(Y, X)
    print(R)
    print(T)

    # 创建一个大小为10的随机向量并把其种的最大值替换为0
    Z = np.random.random(10)
    Z[Z.argmax()] = 0
    print(Z)

    # 创建一个在x轴和y轴上的结构化数组，然后转换到[0,1]*[0,1]的区域
    Z = np.zeros((5, 5), [('x', float), ('y', float)])
    Z['x'], Z['y'] = np.meshgrid(np.linspace(0, 1, 5),
                                 np.linspace(0, 1, 5))
    print(Z)

    # 怎样随机的把一个p元素放入一个2维数组
    n = 10
    p = 3
    Z = np.zeros((n, n))
    np.put(Z, np.random.choice(range(n * n), p, replace=False), 1)
    print(Z)

    # 把一个float数组转化为int数组inplace
    Z = np.arange(10, dtype=np.float32)
    Z = Z.astype(np.int32, copy=False)
    print(Z)

    # 对numpy数组来说，怎样使用enumerate
    Z = np.arange(9).reshape(3, 3)
    for index, value in np.ndenumerate(Z):
        print(index, value)
    for index in np.ndindex(Z.shape):
        print(index, Z[index])

    # 对于一个向量[1,2,3,4,5]，基于它构建一个新的向量，对每个值加入连续的3个0
    Z = np.array([1, 2, 3, 4, 5])
    nz = 3
    Z0 = np.zeros(len(Z) + (len(Z) - 1) * (nz))
    Z0[::nz + 1] = Z
    print(Z0)

    # 交换数组的两行
    A = np.arange(25).reshape(5, 5)
    A[[0, 1]] = A[[1, 0]]
    print(A)

    # 对于一个一维数组Z，构建一个二维数组，第一行为 (Z[0],Z[1],Z[2])，接下来的每一行都移动一格(最后一行为 (Z[-3],Z[-2],Z[-1])
    from numpy.lib import stride_tricks
    def rolling(a, window):
        shape = (a.size - window + 1, window)
        strides = (a.itemsize, a.itemsize)
        return stride_tricks.as_strided(a, shape=shape, strides=strides)

    Z = rolling(np.arange(10), 3)
    print(Z)

    # 给出一个数组C是一个bincount，生成一个新数组A使得 np.bincount(A)==C.
    C = [1, 1, 2, 3, 4, 4, 6]
    A = np.repeat(np.arange(len(C)), C)
    print(A)  # [0,1,2,2,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,6,6]
    print(np.bincount([0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6]))  # 验证
    
    # 对于一个数组Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]， 生成数组R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]
    Z = np.arange(1, 15, dtype=np.uint32)
    R = np.lib.stride_tricks.as_strided(Z, (11, 4), (4, 4))
    print(R)
    
    # 将整数向量转换为矩阵二进制表示形式
    I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
    B = ((I.reshape(-1, 1) & (2 ** np.arange(8))) != 0).astype(int)
    print(B[:, ::-1])
    # Author: Daniel T. McDonald
    I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
    print(np.unpackbits(I[:, np.newaxis], axis=1))


if __name__ == '__main__':
    array_process()
