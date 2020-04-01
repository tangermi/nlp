import scipy.io as sio
import numpy as np


def input_output():
    # 保存 mat 文件
    vect = np.arange(10)
    sio.savemat('array.mat', {'vect': vect})

    # 读取文件
    mat_file_content = sio.loadmat('array.mat')
    print(mat_file_content)

    # 不读取数据到内存的情况下检查MATLAB文件的内容
    mat_file_content = sio.whosmat('array.mat')
    print(mat_file_content)


if __name__ == '__main__':
    input_output()
