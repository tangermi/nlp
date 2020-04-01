# -*- coding:utf-8 -*-
import numpy as np


# 数组排序
def array_sort():
    # 创建一个由随机数组成的10*10的向量，并找出最小和最大值  
    Z = np.random.random((10, 10))
    Zmin, Zmax = Z.min(), Z.max()
    print(Zmin, Zmax)

    # 创建一个向量，它的值为从10到49的整数，然后进行倒序处理
    Z = np.arange(50)
    Z = Z[::-1]
    print(Z)

    # 创建一个大小为10的随机向量，并按升序排序它
    Z = np.random.random(10)
    Z.sort()
    print(Z)

    # 对于每个numpy的scalar格式，print出最小的和最大的可表示值
    X = np.arange(8)
    Y = X + 0.5
    C = 1.0 / np.subtract.outer(X, Y)
    print(np.linalg.det(C))

    # 怎样对数组进行排序，基于第n列的值
    Z = np.random.randint(0, 10, (3, 3))
    print(Z)
    print(Z[Z[:, 1].argsort()])

    # 考虑一组描述三个三角形（带有共享顶点）的10个三元组，找到组成所有三角形的一组唯一线段
    faces = np.random.randint(0, 100, (10, 3))
    F = np.roll(faces.repeat(2, axis=1), -1, axis=1)
    F = F.reshape(len(F) * 3, 2)
    F = np.sort(F, axis=1)
    G = F.view(dtype=[('p0', F.dtype), ('p1', F.dtype)])
    G = np.unique(G)
    print(G)

    # 取到数列的最大的N个值
    Z = np.arange(10000)
    np.random.shuffle(Z)
    n = 5

    # slow
    print(Z[np.argsort(Z)[-n:]])

    # fast
    print(Z[np.argpartition(-Z, n)[:n]])
    
    # 找出数组种最频繁的值
    Z = np.random.randint(0, 10, 50)
    print(np.bincount(Z).argmax())


if __name__ == '__main__':
    array_sort()
