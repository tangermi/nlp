import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def data_visualization():
    df = pd.DataFrame({'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                       'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'], 'age': [42, 52, 36, 24, 73],
                       'female': [0, 1, 1, 0, 1], 'preTestScore': [4, 24, 31, 2, 3],
                       'postTestScore': [25, 94, 57, 62, 70]})
    print(df)

    # 创建preTestScore和postTestScore的散点图，每个点的大小由年龄决定
    # plt.scatter(x=df['age'],y=df['preTestScore'])
    # plt.scatter(x=df['age'],y=df['postTestScore'])
    plt.scatter(x=df['preTestScore'], y=df['postTestScore'], s=df['age'])
    plt.xlabel('preTestScore')
    plt.ylabel('postTestScore')
    plt.title('preTestScore x postTestScore')
    # plt.xticks()
    plt.show()

    # 创建preTestScore和postTestScore的散点图，每个点的大小由年龄决定
    # 这一次的大小为postTestScore的4.5倍，颜色由性别决定
    plt.scatter(x=df['preTestScore'], y=df['postTestScore'], s=df['postTestScore'] * 4.5, c=df['female'])
    plt.xlabel('preTestScore')
    plt.ylabel('postTestScore')
    plt.title('preTestScore x postTestScore')
    plt.show()


if __name__ == '__main__':
    data_visualization()
