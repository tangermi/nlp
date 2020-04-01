# -*- coding:utf-8 -*-
from snownlp import SnowNLP


# 情感分析  输出值越高，代表评价越正面
if __name__ == '__main__':
    text = '这个东西真心很赞'
    s = SnowNLP(text)
    print(s.sentiments)
    text = '这搞鸡毛'
    s = SnowNLP(text)
    print(s.sentiments)
    text = '你也太菜了'
    s = SnowNLP(text)
    print(s.sentiments)
    text = '这个酒店糟透了，地上发霉，床单潮湿，服务差，不推荐'
    s = SnowNLP(text)
    print(s.sentiments)
