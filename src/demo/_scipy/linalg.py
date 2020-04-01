from scipy import linalg
import numpy as np


def linalg_demo():
    # 线性方程组
    # 创建一个numpy数组
    a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
    b = np.array([2, 4, -1])

    # 通过linalg方法计算
    x = linalg.solve(a, b)

    # 打印结果数组
    print(x)

    # 查找一个行列式
    # Declaring the numpy array
    A = np.array([[1, 2], [3, 4]])

    # Passing the values to the det function
    x = linalg.det(A)

    # printing the result
    print(x)

    # 特征值和特征向量
    # Declaring the numpy array
    A = np.array([[1, 2], [3, 4]])

    # Passing the values to the eig function
    l, v = linalg.eig(A)

    # printing the result for eigen values
    print(l)

    # printing the result for eigen vectors
    print(v)

    # 奇异值分解
    # Declaring the numpy array
    a = np.random.randn(3, 2) + 1.j * np.random.randn(3, 2)

    # Passing the values to the eig function
    U, s, Vh = linalg.svd(a)

    # printing the result
    print(U, Vh, s)


if __name__ == '__main__':
    linalg_demo()
