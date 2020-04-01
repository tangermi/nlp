# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def data_group(data_path):
    # 引入数据
    regiment = pd.DataFrame(raw_data, columns=raw_data.keys())
    print(regiment)

    # Nighthawks军团的平均preTestScore是多少
    print(regiment[regiment['regiment'] == 'Nighthawks']['preTestScore'].mean())
    print(regiment.groupby('regiment')['preTestScore'].mean().reset_index().set_index(keys=['regiment']).loc[
              'Nighthawks'])

    # 展示company的数据概览
    print(regiment.groupby('company').describe())

    # 每个company的平均preTestScore
    print(regiment.groupby('company')['preTestScore'].mean())

    # 平均preTestScore对于每个regiment和company的组合
    print(regiment.groupby(['regiment', 'company']).mean()['preTestScore'])

    # 平均preTestScore对于每个regiment和company的组合（不使用金字塔型的序列）
    print(regiment.groupby(['regiment', 'company']).mean()['preTestScore'].unstack())

    # 对于每个regiment和company，Group整个数据
    print(regiment.groupby(['regiment', 'company']).mean())

    # 每个regiment和company有多少条数据
    print(regiment.groupby(['regiment', 'company']).size())

    # 遍历一组并打印该团的名称和整个数据
    for name, group in regiment.groupby('regiment'):
        print(name)
        print(group)


if __name__ == '__main__':
    # 创建一个数据
    # Create an example dataframe
    raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons',
                             'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
                'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd'],
                'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger',
                         'Riani', 'Ali'],
                'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
                'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}

    data_group(raw_data)
