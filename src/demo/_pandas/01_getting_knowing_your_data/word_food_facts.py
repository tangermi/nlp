# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def data_overview(data_path):
    # 引入数据
    food = pd.read_csv(url_local, sep='\t')

    # 显示前25个entries
    print(food.head(5))

    # 数据里有多少个observations 和 columns
    print(food.shape)

    # 打印出数据里所有列的名字
    print(food.columns)

    # 第105列的名字是什么
    print(food.columns[104])

    # 它的类型是什么
    print(food.dtypes['-glucose_100g'])
    print(food.iloc[:, [104]].dtypes)

    # 数据是怎样排序的
    print(food.index)

    # 第19行的Product name是什么
    print(food.at[18, 'product_name'])


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/en.openfoodfacts.org.products.tsv'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/en.openfoodfacts.org.products.tsv'

    data_overview(url_local)   # 数据较大，耐心等待
