# -*- coding:utf-8 -*-
import numpy as np


# 多维数组初始化
def ndarray_initialize():
    # 创建一个3*3的单位矩阵
    Z = np.eye(3)
    print(Z)

    # 创建一个由随机数组成的3*3*3的矩阵
    Z = np.random.random((3, 3, 3))
    print(Z)

    # 创建一个2维数组，边界为1，包含着0
    Z = np.ones((10, 10))
    Z[1:-1, 1:-1] = 0
    print(Z)

    # 创建一个数组并添加边界（由0组成）
    Z = np.ones((5, 5))
    Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
    print(Z)

    # 创建一个5*5的矩阵，对角线下的值为1，2，3，4
    Z = np.diag(1 + np.arange(4), k=-1)
    print(Z)

    # 创建一个8*8的矩阵，令他充满方格图案的数字
    Z = np.zeros((8, 8), dtype=int)
    Z[1::2, ::2] = 1
    Z[::2, 1::2] = 1
    print(Z)

    # 创建一个自定义类型，用来描述颜色，使用4种未指定的字节
    color = np.dtype([("r", np.ubyte, 1),
                      ("g", np.ubyte, 1),
                      ("b", np.ubyte, 1),
                      ("a", np.ubyte, 1)])

    # 创建一个5*5的矩阵，每行的值的范围从0到4
    Z = np.zeros((5, 5))
    Z += np.arange(5)
    print(Z)

    # 创建一个向量，它的值为从0到9的整数，然后把他变为3*3的2维数列
    Z = np.arange(9).reshape(3, 3)
    print(Z)

    # 使用tile方法创建一个8*8的棋盘图案矩阵
    Z = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
    print(Z)

    # 提供2个数组X和Y， 构建一个Cauchy矩阵 C (Cij =1/(xi - yj))
    X = np.arange(8)
    Y = X + 0.5
    C = 1.0 / np.subtract.outer(X, Y)
    print(np.linalg.det(C))

    # 生成一个2维的类高斯阵列
    X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    D = np.sqrt(X * X + Y * Y)
    sigma, mu = 1.0, 0.0
    G = np.exp(-((D - mu) ** 2 / (2.0 * sigma ** 2)))
    print(G)

# 创建一个二维数组的类Z[i,j] == Z[j,i]
class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i, j = index
        super(Symetric, self).__setitem__((i, j), value)
        super(Symetric, self).__setitem__((j, i), value)


def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

def do_symetric():
    # 创建一个二维数组的类Z[i,j] == Z[j,i]
    S = symetric(np.random.randint(0, 10, (5, 5)))
    S[2, 3] = 42
    print(S)

def run():
    for task in tasks:
        eval(task)()
    
tasks = ['ndarray_initialize', 'do_symetric']


if __name__ == "__main__":
    run()
    
