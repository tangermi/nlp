# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def time_series(data_path):
    # 引入数据
    apple = pd.read_csv(data_path, sep=',')

    # 显示前10个entries
    print(apple.head())
    print(apple.dtypes)

    # 把Date列转化为datetime类型
    apple['Date'] = pd.to_datetime(apple['Date'])

    # 把date设为index
    apple.set_index('Date', drop=True, inplace=True)

    # 有重复的日期吗
    print(apple.index.is_unique)

    # 该索引来自最近的日期。将第一个条目设为最早的日期
    apple.sort_index(ascending=True, inplace=True)
    print(apple)

    # 获取每个月的最后一个工作日
    apple_month = apple.resample('BM').mean()
    print(apple_month.head())

    # 第一天和最早的一天之间的天差是多少
    print((apple.index.max() - apple.index.min()).days)

    # 我们有多少个月的数据
    print(len(apple_month.index))

    # plot 'Adj Close'的值，把图片的大小设置为 13.5 * 9 inches
    apl_open = apple['Adj Close'].plot()
    fig = apl_open.get_figure()
    fig.set_size_inches(13.5, 9)
    plt.title('Apple Stock')
    plt.show()


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/09_Time_Series/Apple_Stock/appl_1980_2014.csv'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/appl_1980_2014.csv'

    time_series(url_local)
