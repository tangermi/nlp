from scipy.cluster.vq import kmeans, vq, whiten
from numpy import vstack, array
from numpy.random import rand


def cluster():
    # 数据生成
    data = vstack((rand(100, 3) + array([.5, .5, .5]), rand(100, 3)))

    print(data)

    # 美化数据
    data = whiten(data)
    print(data)

    # 计算三个群集的K均值
    centroids, _ = kmeans(data, 3)

    print(centroids)

    # 将每个值分配给一个集群
    clx, _ = vq(data, centroids)

    # 检查每个观察的聚类
    print(clx)


if __name__ == '__main__':
    cluster()
