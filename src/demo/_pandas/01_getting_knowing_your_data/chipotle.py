# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def data_overview(data_path):
    # 引入数据
    chipo = pd.read_csv(url_local, sep='\t')

    # 显示前10个entries
    print(chipo.head(10))

    # observations的数量
    print(chipo.shape[0])

    # 数据有多少列
    print(chipo.shape[1])

    # 打印出所有列的名字
    print(chipo.columns)

    # 数据是怎样排序的
    print(chipo.index)

    # 数据里哪一样物品被点单的最多
    c = chipo.groupby('item_name')
    c = c.sum()
    c = c.sort_values(['quantity'], ascending=False)
    print(c.head(1))

    # 在choice_description这一列，哪一样物品被点的最多
    c = chipo.groupby('choice_description')
    c = c.sum()
    c = c.sort_values(['quantity'], ascending=False)
    print(c.head(1))

    # 总共有多少个物品被点了
    total_item_ordered = chipo['quantity'].sum()
    print(total_item_ordered)

    # 把价格的格式改成float
    dollarizer = lambda x: float(x[1:-1])
    chipo['item_price'] = chipo['item_price'].apply(dollarizer)
    print(chipo['item_price'].dtype)

    # 这整个期间，数据的流水是多少
    revenue = (chipo['item_price'] * chipo['quantity']).sum()
    print('Revenue was $' + str(np.round(revenue, 2)))

    # 数据里有多少次点单
    orders = chipo['order_id'].value_counts().count()
    print(orders)

    # 每一单的平均流水是多少
    chipo['revenue'] = chipo['item_price'] * chipo['quantity']
    avg_revenue = chipo.groupby('order_id')['revenue'].sum().mean()
    print(avg_revenue)

    # 有多少种不同的物品被卖了出去
    print(chipo['item_name'].value_counts().count())


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/chipotle.tsv'

    data_overview(url_local)
