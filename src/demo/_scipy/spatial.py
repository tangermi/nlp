# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def spatial():
    # Delaunay三角
    points = np.array([[0, 4], [2, 1.1], [1, 3], [1, 2]])
    tri = Delaunay(points)
    plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy())
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()

    # 共面点
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
    tri = Delaunay(points)
    print(tri.coplanar)

    # 凸壳
    points = np.random.rand(10, 2)  # 30 random points in 2-D
    hull = ConvexHull(points)

    plt.plot(points[:, 0], points[:, 1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    plt.show()


if __name__ == '__main__':
    spatial()
