# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def data_visualization(data_path):
    # 引入数据
    online_rt = pd.read_csv(data_path, sep=',', encoding='latin1')

    # 显示前10个entries
    print(online_rt.head(10))
    print(online_rt.info())

    # 绘制一个histogram,表示购买了最多quantity的10个国家，不包含UK
    quantity_rt = online_rt.groupby('Country')['Quantity'].sum()
    print(quantity_rt)
    quantity_rt.drop('United Kingdom').sort_values(ascending=False)[:10].plot(kind='bar')
    plt.xlabel('Countries')
    plt.ylabel('Quantity')
    plt.title('ten countries with most orders')
    # 尝试在pycharm上显示图片
    plt.savefig('/apps/data/ai_nlp_testing/online_retail_hist.png')
    img = Image.open('/apps/data/ai_nlp_testing/online_retail_hist.png')
    img.show()
    plt.show()

    # 移除quantity为负数的行
    online_rt = online_rt[online_rt['Quantity'] >= 0].dropna()

    # 为前3个国家/地区创建一个散点图，其中包含按客户ID的单价数量
    # groupby CustomerID
    customers = online_rt.groupby(['CustomerID', 'Country']).sum()

    # 去掉价格为负数的行
    customers = customers[customers.UnitPrice > 0]

    # get the value of the index and put in the column Country
    customers['Country'] = customers.index.get_level_values(1)

    # top three countries
    top_countries = ['Netherlands', 'EIRE', 'Germany']

    # 筛选出前3个国家
    customers = customers[customers['Country'].isin(top_countries)]

    # 创建FaceGrid
    g = sns.FacetGrid(customers, col="Country")

    # map over a make a scatterplot
    g.map(plt.scatter, "Quantity", "UnitPrice", alpha=1)

    # adds legend
    g.add_legend()
    # 尝试在pycharm上显示图片
    plt.savefig('/apps/data/ai_nlp_testing/online_retail_scatter.png')
    img = Image.open('/apps/data/ai_nlp_testing/online_retail_scatter.png')
    img.show()
    plt.show()

    # plot一个线图，表示每个UnitPrice(x)的revenue(y)
    online_rt['Revenue'] = online_rt.Quantity * online_rt.UnitPrice
    price_start = 0
    price_end = 50
    price_interval = 1
    bucket = np.arange(price_start, price_end, price_interval)

    revenue_per_price = online_rt.groupby(pd.cut(online_rt['UnitPrice'], bucket))['Revenue'].sum()
    print(pd.cut(online_rt['UnitPrice'], bucket))
    print(revenue_per_price.head())
    # plot
    revenue_per_price.plot()
    plt.xlabel('Unit Price (in interval of ' + str(price_interval) + ')')
    plt.ylabel('Revenue')
    # 尝试在pycharm上显示图片
    plt.savefig('/apps/data/ai_nlp_testing/online_retail_plot_1.png')
    img = Image.open('/apps/data/ai_nlp_testing/online_retail_plot_1.png')
    img.show()
    plt.show()
    # 美化
    revenue_per_price.plot()
    plt.xlabel('Unit Price (in interval of ' + str(price_interval) + ')')
    plt.ylabel('Revenue')
    plt.xticks(np.arange(price_start, price_end, 3), np.arange(price_start, price_end, 3))
    plt.yticks([0, 500000, 1000000, 1500000, 2000000, 2500000], ['0', '$0.5M', '$1M', '$1.5M', '$2M', '$2.5M'])
    # 尝试在pycharm上显示图片
    plt.savefig('/apps/data/ai_nlp_testing/online_retail_plot_2.png')
    img = Image.open('/apps/data/ai_nlp_testing/online_retail_plot_2.png')
    img.show()
    plt.show()


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/Online_Retail.csv'

    data_visualization(url_local)
