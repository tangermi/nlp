# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def data_overview(data_path):
    # 引入数据
    users = pd.read_csv(url_local, sep='|', index_col='user_id')

    # 显示前25个entries
    print(users.head(25))

    # 显示最后10个entries
    print(users.tail(10))

    # 数据里有多少个observations
    print(users.shape[0])

    # 数据里有多少列
    print(users.shape[1])

    # 打印出每一列的名字
    print(users.columns)

    # 数据是怎样排序的
    print(users.index)

    # 数据里的数据格式都是怎样的
    print(users.dtypes)

    # 打印出occupation这一列
    print(users['occupation'])

    # 数据里有多少种不同的occupation
    print(users['occupation'].value_counts().count())

    # 数据里最常见的occupation
    print(users['occupation'].value_counts().head(1).index[0])

    # 数据的摘要（默认只对numeric的列生效）
    print(users.describe())

    # 数据的所有列摘要
    print(users.describe(include='all'))

    # occupation这一列的摘要
    print(users['occupation'].describe())

    # 用户的平均年龄是什么
    print(users['age'].mean())

    # 哪个年龄是最少见的
    print(users['age'].value_counts().tail(5))


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u_user'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/u.user'

    data_overview(url_local)
