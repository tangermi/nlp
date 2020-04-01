# -*- coding:utf-8 -*-
from snownlp import SnowNLP


if __name__ == '__main__':
    text = '这个东西真心很赞'
    s = SnowNLP(text)
    # 分词
    print(s.words)
    # 词性标注
    print(s.tags)

