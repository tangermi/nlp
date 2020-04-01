# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def data_stats(data_path):
    # 引入数据
    baby_names = pd.read_csv(data_path, sep=',', header=0)
    print(baby_names.head())

    # 删除两列 'Unnamed: 0' 和 'Id'
    baby_names.drop(['Unnamed: 0', 'Id'], axis=1, inplace=True)

    # 数据集里，男生的名字多还是女生的名字多？
    # print(baby_names.groupby('Gender').agg({'Gender': 'count'}))
    print(baby_names['Gender'].value_counts())

    # 按名称分组数据集并分配给names
    names = baby_names.groupby('Name').sum()

    # 数据集中有多少种不同的名字
    print(len(names))

    # 最常见的名字是什么
    print(names['Count'].idxmax())

    # 多少种不同的名字出现的最少
    # print(names['Count'].sort_values(ascending=True))
    print(len(names[names['Count'] == names['Count'].min()]))

    # 出现次数为中值的名字是什么
    print(names[names['Count'] == names['Count'].median()])

    # 名字出现的方差是多少
    print(names['Count'].std())

    # 数据的摘要，包括最大值，最小值，平均值，方差和四分位数
    print(names['Count'].describe())


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/US_Baby_Names/US_Baby_Names_right.csv'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/US_Baby_Names_right.csv'

    data_stats(url_local)
