# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def data_filter_sort(data_path):
    # 引入数据
    chipo = pd.read_csv(url_local, sep='\t')

    # 把price列转换为float格式
    dollarizer = lambda x: float(x[1:-1])
    chipo['item_price'] = chipo['item_price'].apply(dollarizer)

    # 删除掉重复的数据
    chipo.filtered = chipo.drop_duplicates(['item_name', 'quantity'])

    # 挑选出quantity为1，且价格大于10的产品
    chipo_one_prod = chipo.filtered[chipo['quantity'] == 1]
    print(chipo_one_prod[chipo_one_prod['item_price'] > 10].item_name.unique())

    # 每个物品的价格是多少
    price_per_item = chipo_one_prod[['item_name', 'item_price']]
    price_per_item = price_per_item.sort_values(by='item_price', ascending=False)
    print(price_per_item)

    # 根据商品名字排序
    print(chipo['item_name'].sort_values())

    # 最贵的商品被点了多少次
    most_expensive_item = price_per_item.at[0, 'item_name']
    print(most_expensive_item)
    print(chipo[chipo['item_name'] == most_expensive_item]['quantity'].sum)

    # Veggie Salad Bowl被点了多少次
    print(chipo[chipo['item_name'] == 'Veggie Salad Bowl']['quantity'].sum())

    # 有多少次顾客点了大于一罐的Canned Soda
    chipo_drink_steak_cans = chipo[(chipo['item_name'] == 'Canned Soda') & (chipo['quantity'] > 1)]
    print(len(chipo_drink_steak_cans))


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/chipotle.tsv'

    data_filter_sort(url_local)
