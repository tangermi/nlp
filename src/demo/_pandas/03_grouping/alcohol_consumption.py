# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def data_group(data_path):
    # 引入数据
    drinks = pd.read_csv(url_local, sep=',', header=0)

    print(drinks.head())
    # 哪个大洲平均喝更多啤酒
    beer_sorted = drinks.groupby('continent')['beer_servings'].mean()
    print(beer_sorted)

    # 每个大洲的平均红酒消耗数据
    wine_avg = drinks.groupby('continent')['wine_servings'].describe()
    print(wine_avg)

    # 每个大洲的平均酒精消耗量，在每个种类上的表示
    continent_avg = drinks.groupby('continent').mean()

    # 每个大洲的酒精消耗量中值，在每个种类上的表示
    continent_median = drinks.groupby('continent').median()

    # spirit消耗量的最大值与最小值
    spirit_max = drinks['spirit_servings'].max()
    spirit_min = drinks['spirit_servings'].min()
    print(spirit_max)
    print(spirit_min)
    print(drinks.groupby('continent')['spirit_servings'].agg(['mean', 'min', 'max']))


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/drinks.csv'

    data_group(url_local)
