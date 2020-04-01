# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier


def run():
    # data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # data = np.array(df.iloc[:100, [0, 1, -1]])

    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:,:-1], data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf_sk = KNeighborsClassifier()
    clf_sk.fit(X_train, y_train)

    print(clf_sk.score(X_test, y_test))

if __name__ == "__main__":
    run()
