# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def data_apply(data_path):
    # 引入数据
    df = pd.read_csv(data_path, sep=',', header=0)
    print(df.head())

    # 把数据按照列切割，从school到guardian
    df = df.loc[:, 'school':'guardian']
    print(df)

    # 创建一个lambda方法把字符串首字母大写
    capitalizer = lambda x: x.capitalize()

    # 把 Mjob和Fjob这两列首字母大写
    df['Mjob'] = df['Mjob'].apply(capitalizer)
    df['Fjob'] = df['Fjob'].apply(capitalizer)
    print(df.head())

    # 打印出数据里结尾的元素
    print(df.tail())

    # 创建一个新的方法，名字叫majority，它返回一个布尔值在新的一列称为legal_drinker，条件是年龄是否大于17
    def majority(age):
        if age > 17:
            return True
        else:
            return False

    df['legal_drinker'] = df['age'].apply(majority)
    print(df.head())

    # 将数据里每个数字乘以10
    def times10(x):
        if type(x) is int:
            return x * 10
        else:
            return x

    df = df.applymap(times10)
    print(df.head())


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/student-mat.csv'

    data_apply(url_local)
