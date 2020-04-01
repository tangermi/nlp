# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import datetime as dt


def deleting(data_path):
    # 引入数据
    iris = pd.read_csv(data_path, sep=',', header=None)

    # 显示前10个entries
    print(iris.head())
    iris.info()

    # 为数据集创建列名字
    iris.columns = ['sepal_length','sepal_width','petal_length','petal_width','class']
    
    # 数据集里有缺失数据吗
    print(iris.isnull().sum())
    
    # 将“ petal_length”列的第10到29行的值设置为NaN
    iris.loc[9:29, 'petal_length'] = np.nan
    print(iris.loc[9:29])
    
    # 将NaN值替换为1
    # iris.loc[9:29] = iris.loc[9:29].replace(np.nan,1.0)
    iris.loc[9:29].petal_length.fillna(1, inplace=True)
    
    # 删除掉class这一列
    iris.drop('class', axis=1, inplace=True)
    
    # 把前3行设置为NaN
    iris.loc[:3] = np.nan
    
    # 删掉含有NaN的行
    iris.dropna(inplace=True)
    
    # 重新设置index
    iris.reset_index(drop=True, inplace=True)
    print(iris)
    

if __name__ == '__main__':
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/iris.data'

    deleting(url_local)
