# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def data_filter_sort(data_path):
    # 引入数据
    army = pd.DataFrame(data=raw_data)
    print(army)

    # 把origin这一列设置为index
    army.set_index('origin', inplace=True)

    # 打印出veterans这一列
    print(army['veterans'])

    # 打印出veterans和deaths这2列
    print(army[['veterans', 'deaths']])

    # 打印出每一列的名字
    print(army.columns)

    # 选出缅因和阿拉斯加的deaths, size和deserters这3列
    print(army.loc[['Maine', 'Alaska'], ['deaths', 'size', 'deserters']])

    # 选出第3行到第7行，第3列到第6列的值
    print(army.iloc[2:7, 2:6])

    # 选出第4行之后的每一行
    print(army.iloc[4:, :])

    # 选出前4行
    print(army.iloc[:4, :])

    # 选出第3列到第7列
    print(army.iloc[:, 2:7])

    # 选出deaths大于50的行
    print(army[army['deaths'] > 50])

    # 选出deaths大于500或者小于50的行
    print(army[(army['deaths'] > 500) | (army['deaths'] < 50)])

    # 选出所有命名不是Dragoons的军团
    print(army[army['regiment'] != 'Dragoons'])

    # 选出来自Texas和Arizona的行
    print(army.loc[['Texas', 'Arizona'], :])

    # 选出来自Arizona这一行的第三个格子
    print(army.loc[['Arizona']].iloc[:, 2])

    # 选出deaths这一列的第三个格子
    print(army.loc[:, ['deaths']].iloc[2])


if __name__ == '__main__':
    # 创建一个数据
    # Create an example dataframe about a fictional army
    raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
                'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
                'deaths': [523, 52, 25, 616, 43, 234, 523, 62, 62, 73, 37, 35],
                'battles': [5, 42, 2, 2, 4, 7, 8, 3, 4, 7, 8, 9],
                'size': [1045, 957, 1099, 1400, 1592, 1006, 987, 849, 973, 1005, 1099, 1523],
                'veterans': [1, 5, 62, 26, 73, 37, 949, 48, 48, 435, 63, 345],
                'readiness': [1, 2, 3, 3, 2, 1, 2, 3, 2, 1, 2, 3],
                'armored': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
                'deserters': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
                'origin': ['Arizona', 'California', 'Texas', 'Florida', 'Maine', 'Iowa', 'Alaska', 'Washington', 'Oregon', 'Wyoming', 'Louisana', 'Georgia']}

    data_filter_sort(raw_data)
