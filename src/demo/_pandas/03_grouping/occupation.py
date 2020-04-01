# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def data_group(data_path):
    # 引入数据
    users = pd.read_csv(url_local, sep='|', header=0)

    print(users.head())

    # 每个职位的平均年龄
    age_avg = users.groupby('occupation')['age'].mean()
    print(age_avg)

    # 每个职位的男性比例，以从大到小排列
    def gender_to_numeric(x):
        if x == 'M':
            return 1
        if x == 'F':
            return 0

    users['gender_n'] = users['gender'].apply(gender_to_numeric)
    male_ratio = users.groupby('occupation')['gender_n'].sum() / users['occupation'].value_counts() * 100
    print(male_ratio.sort_values(ascending=False))

    # 找出每个职位的最大年龄与最小年龄
    print(users.groupby('occupation')['age'].agg(['max', 'min']))

    # 对于每个性别和职位的组合，计算平均年龄
    print(users.groupby(['gender', 'occupation'])['age'].mean())

    # 对于每个职位，展示男女性别占比
    gender_ocup = users.groupby(['occupation', 'gender']).agg({'gender': 'count'})
    print(gender_ocup)
    ocup_count = users.groupby('occupation').agg('count')
    ocup_gender = gender_ocup.div(ocup_count, level='occupation') * 100
    print(ocup_gender.loc[:, 'gender'])


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u_user'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/u.user'

    data_group(url_local)
