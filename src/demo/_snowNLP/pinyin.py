# -*- coding:utf-8 -*-
from snownlp import SnowNLP


if __name__ == '__main__':
    text = '这个东西真心很赞'
    s = SnowNLP(text)
    print(s.pinyin)