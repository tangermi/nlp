import scipy.integrate
from numpy import exp
from math import sqrt


def integrate():
    # 单积分
    f = lambda x: exp(-x ** 2)
    i = scipy.integrate.quad(f, 0, 1)  # 四元函数返回两个值，其中第一个数字是积分值，第二个数值是积分值绝对误差的估计值
    print(i)

    # 双重积分
    f = lambda x, y: 16 * x * y
    g = lambda x: 0
    h = lambda y: sqrt(1 - 4 * y ** 2)
    i = scipy.integrate.dblquad(f, 0, 0.5, g, h)
    print(i)


if __name__ == '__main__':
    integrate()
