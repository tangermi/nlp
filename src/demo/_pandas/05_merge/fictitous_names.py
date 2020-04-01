# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def data_merge(raw_data_1, raw_data_2, raw_data_3):
    # 引入数据
    data1 = pd.DataFrame(raw_data_1)
    data2 = pd.DataFrame(raw_data_2)
    data3 = pd.DataFrame(raw_data_3)

    # 根据行合并前2个数据集并赋值给all_data
    all_data = data1.append(data2)
    print(all_data)

    # 根据列合并前2个数据集并赋值给all_data_col
    all_data_col = pd.concat([data1, data2], axis=1)
    print(all_data_col)

    # 打印data3
    print(data3)

    # 根据subject_id的值合并all_data和data3
    print(pd.merge(all_data, data3, on='subject_id'))

    # 合并data1和data2里有共同subject_id的部分
    print(pd.merge(data1, data2, on='subject_id'))

    # 合并data1和data2里所有的值，并在可能的情况下合并双方的匹配记录
    print(pd.merge(data1, data2, on='subject_id', how='outer'))


if __name__ == '__main__':
    raw_data_1 = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}

    raw_data_2 = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}

    raw_data_3 = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}

    data_merge(raw_data_1, raw_data_2, raw_data_3)
