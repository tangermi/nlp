# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def data_visualization(data_path):
    # 引入数据
    chipo = pd.read_csv(data_path, sep='\t')

    # 显示前10个entries
    print(chipo.head(10))

    # 为最畅销的5个物品绘制一个histogram
    top_sellings = chipo.groupby('item_name')['quantity'].sum().sort_values(ascending=False)
    print(top_sellings)
    top_sellings[:5].plot(kind='bar')
    # Set the title and labels
    plt.xlabel('Items')
    plt.ylabel('Ordered')
    plt.title('Most ordered Chipotle\'s Items')
    plt.show()
    
    # 绘制一个散点图，表示每单价格和每单物品个数的关系
    dollarizer = lambda x: float(x[1:])
    chipo['item_price'] = chipo['item_price'].apply(dollarizer)
    price_item = chipo.groupby('order_id').sum()
    print(price_item)
    plt.scatter(x=price_item['quantity'], y=price_item['item_price'])
    plt.xlabel('Order Price')
    plt.ylabel('Items ordered')
    plt.title('Number of items ordered per order price')
    # 尝试在pycharm上显示图片
    plt.savefig('/apps/data/ai_nlp_testing/price2item.png')
    img = Image.open('/apps/data/ai_nlp_testing/price2item.png')
    plt.imshow(img)


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
    url_local = '/apps/data/ai_nlp_testing/raw/pandas_exercises/chipotle.tsv'

    data_visualization(url_local)
