# 定义分词函数preprocess_text
# 参数content_lines即为上面转换的list
# 参数sentences是定义的空list，用来储存分词后的数据
import random
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlibp.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import gensim
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
import multiprocessing

sentences = []

sentences = ['时间 问你 我们 群殴', '大家 文献 二次 去啊', '物品 你的 我的 开心']
random.shuffle(sentences)

# 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
# 统计每个词语的tf-idf权值
transformer = TfidfTransformer()
# 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(sentences))
# 获取词袋模型中的所有词语
word = vectorizer.get_feature_names()
# 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()
# 查看特征大小
print('Features length: ' + str(len(word)))
print(word)

numClass = 3  # 聚类分几簇
clf = KMeans(n_clusters=numClass, max_iter=10000, init="k-means++", tol=1e-6)  # 这里也可以选择随机初始化init="random"
pca = PCA(n_components=2)  # 降维
TnewData = pca.fit_transform(weight)  # 载入N维，权重
s = clf.fit(TnewData)


def plot_cluster(result, newData, numClass):
    plt.figure(2)
    Lab = [[] for i in range(numClass)]
    index = 0
    for labi in result:
        Lab[labi].append(index)
        index += 1
    color = ['oy', 'ob', 'og', 'cs', 'ms', 'bs', 'ks', 'ys', 'yv', 'mv', 'bv', 'kv', 'gv', 'y^', 'm^', 'b^', 'k^',
             'g^'] * 3
    for i in range(numClass):
        x1 = []
        y1 = []
        for ind1 in newData[Lab[i]]:
            # print ind1
            try:
                y1.append(ind1[1])
                x1.append(ind1[0])
            except:
                pass
        plt.plot(x1, y1, color[i])

    # 绘制初始中心点
    x1 = []
    y1 = []
    for ind1 in clf.cluster_centers_:
        try:
            y1.append(ind1[1])
            x1.append(ind1[0])
        except:
            pass
    plt.plot(x1, y1, "rv")  # 绘制中心
    plt.show()

# pca = PCA(n_components=2)  # 输出两维，平面可视化
# newData = pca.fit_transform(weight)  # 载入N维
# result = list(clf.predict(newData))
# plot_cluster(result, newData, numClass)

from sklearn.manifold import TSNE
ts =TSNE(2)
newData = ts.fit_transform(weight)
result = list(clf.predict(TnewData))
plot_cluster(result,newData,numClass)