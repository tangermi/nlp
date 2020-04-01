# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def data_visualization(data_path):
    # 引入数据
    tips = pd.read_csv(data_path, sep=',')

    # 显示前10个entries
    print(tips.head(10))

    # 删掉 Unnamed: 0 这一列
    del tips['Unnamed: 0']
    print(tips.head())

    # 用histogram的形式plot total_bill这一列
    ttbill = sns.distplot(tips['total_bill'])
    ttbill.set(xlabel='Value', ylabel='Frequency', title='Total Bill')
    sns.despine()
    plt.show()

    # 绘制一个散点图描绘tip和total_bill之间的关系
    # plt.scatter(x=tips['total_bill'],y=tips['tip'])
    # plt.title('total_bill x tip')
    # plt.xlabel('total_bill')
    # plt.ylabel('tip')
    sns.jointplot(x='total_bill', y='tip', data=tips)
    plt.show()

    # 绘制一张图描绘tip，size和total_bill之间的关系
    # plt.scatter(x=tips['total_bill'],y=tips['tip'],s=tips['size']*10)
    # plt.title('total_bill x tip')
    # plt.xlabel('total_bill')
    # plt.ylabel('tip')
    sns.pairplot(tips)
    plt.show()

    # 表示天气与total_bill值之间的关系
    sns.stripplot(x='day', y='total_bill', data=tips, jitter=True)
    plt.show()

    # 创建一个散点图，以day为y轴，tip为x轴，按性别区分点
    # plt.scatter(x=tips['tip'],y=tips['day'],c=tips['sex'].map({'Female':0,'Male':1}))
    # plt.title('tip x day')
    # plt.xlabel('tip')
    # plt.ylabel('day')
    sns.stripplot(x='tip', y='day', hue='sex', data=tips, jitter=True)
    plt.show()

    # 创建一个箱形图，显示每天的total_bill，把时间区分开（晚餐或午餐）
    sns.boxplot(x='day', y='total_bill', hue='time', data=tips)
    plt.show()

    # 为晚餐和午餐创建两个tip的直方图。他们必须对齐
    sns.set(style='ticks')
    g = sns.FacetGrid(tips, col='time')
    g.map(plt.hist, 'tip')
    plt.show()

    # 创建两个散点图，一个用于男性，另一个用于女性，显示total_bill值和小费关系，因吸烟者或不吸烟者而异
    sns.set(style='ticks')
    g = sns.FacetGrid(tips, col='sex', hue='smoker')
    g.map(plt.scatter, 'total_bill', 'tip', alpha=.7)
    g.add_legend()
    plt.show()


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Tips/tips.csv'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/tips.csv'

    data_visualization(url_local)
