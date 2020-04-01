# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def data_filter_sort(data_path):
    # 引入数据
    euro12 = pd.read_csv(url_local, sep=',')

    goal = euro12['Goals']
    print(goal)

    # 多少只队伍参与了Euro2012
    euro12.shape[0]

    # 数据包含了多少列
    euro12.shape[1]

    # 把列Team, Yellow Cards和Red Cards组成一个 dataframe，叫做discipline
    discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
    print(discipline)

    # 根据 Red Cards, Yellow Cards对Team进行排序
    print(discipline.sort_values(['Red Cards', 'Yellow Cards'], ascending=False))

    # 计算队伍们获得黄牌的平均值
    print(discipline['Yellow Cards'].mean())

    # 筛选出进球出超过6个的队伍
    print(euro12[euro12['Goals'] > 6])

    # 筛选出名字以G开头的队伍
    print(euro12[euro12['Team'].str.startswith('G')])

    # 挑选出靠前的7列
    print(euro12.iloc[:, :7])

    # 选出所有的列除了最后3列
    print(euro12.iloc[:, :-3])

    # 展示来自英格兰，意大利和俄国的Shooting Accuracy
    print(euro12[euro12['Team'].isin(['England', 'Italy', 'Russia'])][['Team', 'Shooting Accuracy']])


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/Euro_2012_stats_TEAM.csv'

    data_filter_sort(url_local)
