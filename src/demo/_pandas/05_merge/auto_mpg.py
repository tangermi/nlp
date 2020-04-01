# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def data_merge(data_path, data_path2):
    # 引入数据
    car1 = pd.read_csv(data_path, sep=',', header=0)
    car2 = pd.read_csv(data_path2, sep=',', header=0)
    print(car1.head())
    print(car2.head())

    # 第一个数据集里有一些没有命名的空列，修复这个问题
    car1 = car1.loc[:, 'mpg':'car']
    print(car1.head())

    # 两个数据集各有多少个observations
    print(car1.shape[0], car2.shape[0])

    # 合并两个数据集，赋值给一个变量cars
    cars = car1.append(car2)
    print(cars)

    # 这里有一列缺失，创建一个随机数series，范围从15000到73000
    nr_owners = np.random.randint(15000, 73000, size = 398, dtype='l')
    print(nr_owners)

    # 把owners加入到数据集cars
    cars['owners'] = nr_owners
    print(cars.tail())


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars1.csv'
    url2 = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars2.csv'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/cars1.csv'
    url_local2 = '/apps/data/ai_nlp_testing/raw/pandas_exercises/cars2.csv'

    data_merge(url_local, url_local2)
