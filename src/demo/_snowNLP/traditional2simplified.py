# -*- coding:utf-8 -*-
from snownlp import SnowNLP


# 繁体中文转换为简体中文
if __name__ == '__main__':
    text = u'「繁體字」「繁體中文」的叫法在臺灣亦很常見。'
    s = SnowNLP(text)
    print(s.han)