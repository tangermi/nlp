# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def data_merge():
    # 生成3组随机的series，每个的长度为100。第一个的取值范围是1-4. 第二个的取值范围是1-3. 第三个的取值范围是10000到30000
    s1 = pd.Series(np.random.randint(1, 4, 100))
    s2 = pd.Series(np.random.randint(1, 3, 100))
    s3 = pd.Series(np.random.randint(10000, 30000, 100))

    # 将series合并来构建一个data frame
    housemkt = pd.concat([s1, s2, s3], axis=1)
    print(housemkt)

    # 将列的名字修改为bedrs, bathrs, price_sqr_meter
    housemkt.columns = ['bedrs', 'bathrs', 'price_sqr_meter']
    print(housemkt)

    # 使用3个series的值创建一列DataFrame并将其分配给'bigcolumn'
    bigcolumn = pd.concat([s1, s2, s3], axis=0)
    print(bigcolumn)

    # index只到达了99，是真的吗
    print(len(bigcolumn))

    # 重新设置index，从0到299
    bigcolumn.reset_index(drop=True, inplace=True)
    print(bigcolumn)


if __name__ == '__main__':
    data_merge()
