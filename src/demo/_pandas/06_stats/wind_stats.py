# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import datetime


def data_stats(data_path):
    # 引入数据,将前3列替换为datatime格式
    data = pd.read_csv(data_path, sep='\s+', header=0, parse_dates=[[0, 1, 2]])
    print(data.head())

    # 修复日期里显示2061年的问题
    def fix_date(x):
        year = x.year - 100 if x.year > 1989 else x.year
        return datetime.date(year, x.month, x.day)

    data['Yr_Mo_Dy'] = data['Yr_Mo_Dy'].apply(fix_date)
    print((data.head()))

    # 把日期设为index，类型为datetime64[ns]
    data['Yr_Mo_Dy'] = pd.to_datetime(data['Yr_Mo_Dy'])
    data.set_index('Yr_Mo_Dy', inplace=True, drop=True)
    print(data.head())

    # 计算数据集里每个地区有多少缺失值
    print(data.isnull().sum())

    # 计算数据里总共有多少非缺失值
    print(data.shape[0] - data.isnull().sum())
    print(data.notnull().sum().sum())

    # 计算所有地区所有时间的平均风速
    print(data.fillna(0).values.flatten().mean())

    # 创建一个DataFrame分配给loc_stats，计算每个地区的最小，最大，平均风速和方差。
    loc_stats = data.agg(['min', 'max', 'mean', 'std'])
    print(loc_stats)

    # 创建一个DataFrame分配给day_stats，计算每天的最小，最大，平均风速和方差。
    day_stats = data.agg(['min', 'max', 'mean', 'std'], axis=1)
    print(day_stats)

    # 找出每个地区一月份的平均风速
    print(data.loc[data.index.month == 1].mean())

    # 把数据集缩小为以年为单位
    print(data.resample('1Y').mean())
    print(data.groupby(data.index.to_period('A')).mean())

    # 把数据缩小为以月为单位
    print(data.resample('1M').mean())
    print(data.groupby(data.index.to_period('M')).mean())

    # 把数据缩小为以周为单位
    print(data.resample('1W').mean())
    print(data.groupby(data.index.to_period('W')).mean())

    # 计算每周的平均，最小，最大风速和方差（第一周开始于1961.Jan.2）
    weekly = data.groupby(data.index.to_period('W')).agg(['mean', 'min', 'max', 'std'])
    print(weekly.iloc[1:])


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/Wind_Stats/wind.data'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/wind.data'

    data_stats(url_local)
