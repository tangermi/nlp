# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def data_visualization(data_path):
    # 引入数据
    titanic = pd.read_csv(data_path, sep=',')

    # 显示前10个entries
    print(titanic.head(10))
    titanic.set_index('PassengerId', drop=True, inplace=True)

    # 绘制一个饼形图表示男女比例
    males = (titanic['Sex'] == 'male').sum()
    females = (titanic['Sex'] == 'female').sum()
    proportions = [males, females]
    plt.pie(proportions, labels=['male', 'female'], colors=['blue', 'red'], startangle=90, shadow=False,
            explode=(0.15, 0), autopct='%1.1f%%')
    plt.show()

    # 使用已付费用和年龄创建散点图，按性别区分地块颜色
    binsval = np.arange(0, 100, 10)
    lm = sns.lmplot(x='Age', y='Fare', hue='Sex', data=titanic, fit_reg=False)
    lm.set(title='Fare x Age')
    axes = lm.axes
    axes[0, 0].set_ylim(-5, )
    axes[0, 0].set_xlim(-5, 85)
    plt.show()

    # 有多少人幸存
    survived = titanic['Survived'].sum()
    print(survived)

    # 用已付费用创建直方图
    binsval = np.arange(0, 600, 10)
    plt.hist(x=titanic.Fare, bins=binsval)
    plt.xlabel = 'Fare payed'
    plt.ylabel = 'Frequency'
    plt.title = 'Fare Payed Histrogram'
    plt.show()
    fare_dist = sns.distplot(titanic.Fare, bins=binsval)
    fare_dist.set(title='Fare Payed Histogram', xlabel='Fare payed', ylabel='Frequency')
    plt.show()


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Titanic_Desaster/train.csv'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/train.csv'

    data_visualization(url_local)
