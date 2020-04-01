# -*- coding:utf-8 -*-
import numpy as np


def numpy_features():
    # 打印numpy的版本以及配置
    print(np.__version__)
    np.show_config()

    # 命令行找到numpy下add方法的文档
    # %run `python -c "import numpy; numpy.info(numpy.add)"`

    # 下面输出的结果是什么
    print(0 * np.nan)
    print(np.nan == np.nan)
    print(np.inf > np.nan)
    print(np.nan - np.nan)
    print(np.nan in set([np.nan]))
    print(0.3 == 3 * 0.1)

    # 对于一个大小为6*7*8的数组，它的第100个元素在哪个位置
    print(np.unravel_index(99, (6, 7, 8)))

    # 下面输出结果是什么
    print(sum(range(5), -1))
    from numpy import sum as np_sum
    print(np_sum(range(5), -1))

    # 对于一个整数向量Z， 下面哪一个表达式是合理的
    # Z = [1,2]
    # Z**Z
    # 2 << Z >> 2
    # Z <- Z
    # 1j*Z
    # Z/1/1
    # Z<Z>Z

    # 下面的输出结果是什么
    print(np.array(0) / np.array(0))
    print(np.array(0) // np.array(0))
    print(np.array([np.nan]).astype(int).astype(float))

    # 怎样禁用所有numpy的警告（不推荐）
    # Suicide mode on
    defaults = np.seterr(all="ignore")
    Z = np.ones(1) / 0

    # Back to sanity
    _ = np.seterr(**defaults)

    # 一个可以达到同样效果的方法

    with np.errstate(divide='ignore'):
        Z = np.ones(1) / 0

    # 下面的表达式是True吗
    np.sqrt(-1) == np.emath.sqrt(-1)

    # 怎样计算((A+B)*(-A/2))，在不赋值给新变量的情况下
    A = np.ones(3) * 1
    B = np.ones(3) * 2
    C = np.ones(3) * 3
    np.add(A, B, out=B)
    np.divide(A, 2, out=A)
    np.negative(A, out=A)
    np.multiply(A, B, out=A)

    # 怎样取得昨天，今天和明天的日期
    yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
    today = np.datetime64('today', 'D')
    tomorrow = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
    print(yesterday, today, tomorrow)

    # 怎样取到2016年8月的所有日期
    Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
    print(Z)

    # 使一个数列不可修改（只读）
    Z = np.zeros(10)
    Z.flags.writeable = False
    # Z[0] = 1

    # 打印一个数组的所有值（IDE默认使用...省略）
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    Z = np.zeros((16, 16))
    print(Z)

    # 怎样读取以下文件
    from io import StringIO
    s = StringIO("""1, 2, 3, 4, 5\n
                    6,  ,  , 7, 8\n
                    ,  , 9,10,11\n""")  # Fake file
    Z = np.genfromtxt(s, delimiter=",", dtype=np.int)
    print(Z)
    
    # 对于2个向量A，B，写出inner, outer, sum和mul方法的einsum equivalent
    A = np.random.uniform(0, 1, 10)
    B = np.random.uniform(0, 1, 10)

    np.einsum('i->', A)  # np.sum(A)
    np.einsum('i,i->i', A, B)  # A * B
    np.einsum('i,i', A, B)  # np.inner(A, B)
    np.einsum('i,j->ij', A, B)  # np.outer(A, B)


if __name__ == '__main__':
    numpy_features()
