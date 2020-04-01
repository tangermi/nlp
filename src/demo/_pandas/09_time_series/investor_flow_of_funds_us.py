# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import datetime as dt


def time_series(data_path):
    # 引入数据
    df = pd.read_csv(data_path, sep=',', header=0)

    # 显示前10个entries
    print(df.head())
    df.info()

    # 把Date列设置为index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', drop=True, inplace=True)
    print(df)

    # 将数据的频率更改为每月一次，将值求和并将其分配给每月一次。
    monthly = df.resample('M').sum()
    print(monthly)

    # 它使用没有NaN数据的月份填充了dataFrame。让我们删除这些行
    monthly = monthly.replace(0, np.nan)
    monthly = monthly.dropna(how='all', axis=0)
    monthly = monthly.replace(np.nan, 0)
    print(monthly)

    # 把频率改成以年为单位
    yearly = monthly.resample('Y').sum()
    print(yearly)


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/master/data/weekly.csv'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/weekly.csv'

    time_series(url_local)
