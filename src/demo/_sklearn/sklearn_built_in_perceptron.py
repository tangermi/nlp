# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt


def run():
    # load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target

    # 命名features
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    df.label.value_counts()

    data = np.array(df.iloc[:100, [0, 1, -1]])

    X, y = data[:, :-1], data[:, -1]

    y = np.array([1 if i == 1 else -1 for i in y])

    clf = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)
    clf.fit(X, y)

    # Weights assigned to the features.
    print(clf.coef_)

    # 截距 Constants in decision function.
    print(clf.intercept_)

    x_ponits = np.arange(4, 8)
    y_ = -(clf.coef_[0][0] * x_ponits + clf.intercept_) / clf.coef_[0][1]
    plt.plot(x_ponits, y_)

    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    run()
