# -*- coding: UTF-8 -*-
import math
import numpy as np
import pandas as pd


# 卡方检查计算
class ChiSquare:
    def get_variance(self, df):
        row_count = df.shape[0] - 1
        col_count = df.shape[1] - 1
        v = (row_count - 1) * (col_count - 1)
        return v

    # 转为矩阵求卡方距离z`
    def get_chi_square_value(self, df1, df2):
        mtr1 = df1.astype(int).as_matrix()
        mtr2 = df2.astype(int).as_matrix()
        mtr = ((mtr1 - mtr2) ** 2) / mtr2
        return mtr.sum()

    # 分类频数
    def get_classification(self, df):
        df['row_total'] = df.sum(axis=1)
        df.loc['col_total'] = df.sum(axis=0)
        df2 = df.copy()
        total = df2[['row_total']].loc[['col_total']].values[0][0]
        for col in df2:
            df2[col] = df2[[col]].loc[['col_total']].values[0][0] * df2['row_total'] / total
        df2 = df2.drop(['col_total'])
        df2.loc['col_total'] = df2.sum(axis=0)
        x2 = self.get_chi_square_value(df, df2)  # 顺序：(实际df,推算df)
        v = self.get_variance(df2)  # v=（行数-1）（列数-1）

        return x2, v

    # 显示结果
    def display(self, x2, v):
        print("卡方值：χ2 = %s" % x2)
        print("自由度：v = %s" % v)


def run():
    df = pd.DataFrame({
        '属于娱乐': [19, 34],
        '不属于娱乐': [24, 10]
    })
    print(df.shape)
    cs = ChiSquare()
    x2, v = cs.get_classification(df)
    cs.display(x2, v)


if __name__ == "__main__":
    run()
