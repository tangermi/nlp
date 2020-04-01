import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


def interpolate():
    # 创建一些数据
    x = np.linspace(0, 4, 12)
    y = np.cos(x ** 2 / 3 + 4)
    print(x, y)

    # 现在，有两个数组。 假设这两个数组作为空间点的两个维度，使用下面的程序进行绘图，并看看它们的样子
    plt.plot(x, y, 'o')
    plt.show()

    # 一维插值
    f1 = interp1d(x, y, kind='linear')

    f2 = interp1d(x, y, kind='cubic')

    # 创建更多长度的新输入以查看插值的明显区别
    xnew = np.linspace(0, 4, 30)

    plt.plot(x, y, 'o', xnew, f1(xnew), '-', xnew, f2(xnew), '--')

    plt.legend(['data', 'linear', 'cubic', 'nearest'], loc='best')

    plt.show()

    # 样条曲线
    x = np.linspace(-3, 3, 50)
    y = np.exp(-x ** 2) + 0.1 * np.random.randn(50)
    plt.plot(x, y, 'ro', ms=5)
    # plt.show()

    # 使用平滑参数的默认值。效果如下
    spl = UnivariateSpline(x, y)
    xs = np.linspace(-3, 3, 1000)
    plt.plot(xs, spl(xs), 'g', lw=3)
    # plt.show()

    # 手动更改平滑量。效果如下
    spl = UnivariateSpline(x, y)
    xs = np.linspace(-3, 3, 1000)
    plt.plot(xs, spl(xs), 'g', lw=3)
    # plt.show()

    spl.set_smoothing_factor(0.5)
    plt.plot(xs, spl(xs), 'b', lw=3)
    plt.show()


if __name__ == '__main__':
    interpolate()
