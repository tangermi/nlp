# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import datetime as dt


def deleting(data_path):
    # 引入数据
    wine = pd.read_csv(data_path, sep=',', header=None, )

    # 显示前10个entries
    print(wine.head(10))

    # 删除第一，第四，第七，第九，第十一，第十三和第十四列
    print(wine.columns[[0, 3, 6, 8, 10, 12, 13]])
    wine.drop(wine.columns[[0, 3, 6, 8, 10, 12, 13]], axis=1, inplace=True)
    print(wine)

    # 把列的名字设置为1) alcohol    2) malic_acid   3) alcalinity_of_ash    4) magnesium    5) flavanoids   6) proanthocyanins
    wine.columns = ['alcohol', 'malic_acid', 'alcalinity_of_ash', 'magnesium', 'flavanoids', 'proanthocyanins', 'hue']

    # 把alcohol前3行的值设置为NaN
    wine.loc[[0, 1, 2], 'alcohol'] = np.nan

    # 把magnesium第3行和第4行的值设置为NaN
    wine.loc[[2, 3], 'magnesium'] = np.nan
    print(wine.head())

    # 把alcohol里的NaN替换为10，把magnesium的替换为100
    wine.alcohol.fillna(10, inplace=True)
    wine.magnesium.fillna(100, inplace=True)

    # 计算缺失值的数量
    print(wine.isnull().sum())

    # 创建一个大小为10的随机数组，最大值不超过10
    random_array = np.random.randint(0, 10, 10)
    print(random_array)

    # 使用您生成的随机数作为索引，并为每个单元格分配NaN值
    wine.alcohol[random_array] = np.nan
    wine.head()

    # 现在我们都多少缺失值 
    print(wine.isnull().sum())

    # 删除掉含有缺失值的行
    wine.dropna(axis=0, how='any', inplace=True)
    print(wine.head())

    # 仅打印alcohol列中的非空值
    print(wine['alcohol'].dropna())

    # 重新设置排序
    wine.reset_index(drop=True, inplace=True)
    print(wine)


if __name__ == '__main__':
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/wine.data'

    deleting(url_local)
