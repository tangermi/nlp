# -*- coding:utf-8 -*-
import numpy as np


# 一维向量初始化
def vector_initialize():
    # 创建一个大小为10的空向量
    Z = np.zeros(10)
    print(Z)

    # 找到数组的存储大小
    Z = np.zeros((10, 10))
    print("%d bytes" % (Z.size * Z.itemsize))

    # 创建一个大小为10的空向量，并把它的第5个元素赋值为1
    Z = np.zeros(10)
    Z[4] = 1
    print(Z)

    # 创建一个向量，它的值为从10到49的整数
    Z = np.arange(10, 50)
    print(Z)

    # 找到非0元素的序列
    nz = np.nonzero([1, 2, 0, 0, 4, 0])
    print(nz)

    # 创建一个大小为30的随机向量并找出平均值
    Z = np.random.random(30)
    m = Z.mean()
    print(m)

    # 创建一个大小为10的向量，值的范围从0到1（不包含1）
    Z = np.linspace(0, 1, 11, endpoint=False)[1:]
    print(Z)

    # 一个生成器可以生成10个整数去构建一个数组
    def generate():
        for x in range(10):
            yield x

    Z = np.fromiter(generate(), dtype=float, count=-1)
    print(Z)


if __name__ == "__main__":
    vector_initialize()
