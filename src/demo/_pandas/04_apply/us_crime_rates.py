# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def data_apply(data_path):
    # 引入数据
    crime = pd.read_csv(data_path, sep=',', header=0)
    print(crime.head())

    # 数据里每一列的类型都是什么
    print(crime.info())

    # 把Year这一列的类型修改成datetime64
    crime['Year'] = pd.to_datetime(crime['Year'], format='%Y')
    print(crime.info())

    # 把Year这一列设置为index
    crime = crime.set_index(keys='Year', drop=True)

    # 删除掉Total这一列
    crime = crime.drop('Total', axis=1)

    # 根据decades来group数据，得到相加值
    crime_decades = crime.resample('10AS').sum()
    population = crime['Population'].resample('10AS').max()
    crime_decades['Population'] = population
    print(crime_decades)

    # 生活在美国，哪一个世纪是最危险的
    print(crime_decades.idxmax())


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/US_Crime_Rates/US_Crime_Rates_1960_2014.csv'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/US_Crime_Rates_1960_2014.csv'

    data_apply(url_local)
