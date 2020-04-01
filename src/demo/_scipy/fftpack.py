from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.fftpack import fftfreq
from scipy.fftpack import dct, idct
import numpy as np
import math


def fftpack():
    # create an array with random n numbers
    x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])

    # 一维离散傅立叶变换
    y = fft(x)
    print(y)

    # 逆傅里叶变换
    yinv = ifft(y)
    print(yinv)

    # 一个(嘈杂的)输入信号可能看起来如下 
    time_step = 0.02
    period = 5.
    time_vec = np.arange(0, 20, time_step)
    sig = np.sin(2 * np.pi / period * time_vec) + 0.5 * np.random.randn(time_vec.size)
    print(sig.size)

    # scipy.fftpack.fftfreq()函数将生成采样频率，scipy.fftpack.fft()将计算快速傅里叶变换。
    sample_freq = fftfreq(sig.size, d=time_step)
    sig_fft = fft(sig)
    print(sample_freq)
    print(sig_fft)

    # 离散余弦变换
    mydict = dct(np.array([4., 3., 5., 10., 5., 3.]))
    print(mydict)
    # 逆离散余弦变换
    d = idct(np.array([4., 3., 5., 10., 5., 3.]))
    print(d)


if __name__ == '__main__':
    fftpack()
