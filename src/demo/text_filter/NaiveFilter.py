# -*- coding:utf-8 -*-

from collections import defaultdict

import re


# 一个简单的过滤，把敏感词从文本中移除

class NaiveFilter():
    def __init__(self):
        self.keywords = set([])

    def parse(self, path):
        for keyword in open(path, 'r', encoding='utf-8'):
            self.keywords.add(keyword.strip().lower())

    def filter(self, message, repl='*'):
        message = str(message).lower()
        for kw in self.keywords:
            message = message.replace(kw, repl)
        return message





if __name__ == "__main__":
    gfw = NaiveFilter()
    gfw.parse(r"/apps/data/ai_nlp_testing/raw/sensored_words/keywords")
    import time
    t = time.time()
    print(gfw.filter("法轮功 shabi", "*"))
    print(gfw.filter("针孔摄像机 caonima", "*"))
    print(gfw.filter("售假人民币 操你妈", "*"))
    print(gfw.filter("传世私服 操场我操操操", "*"))
    print(gfw.filter("您吃了没", "*"))
    print(time.time() - t)


    test_first_character()