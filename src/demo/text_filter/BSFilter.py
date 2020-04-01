# -*- coding:utf-8 -*-

from collections import defaultdict

import re

# 同样是过滤敏感词，使用 back sorted mapping方法来减少替换次数

class BSFilter():
    def __init__(self):
        self.keywords = []
        self.kwsets = set([])
        self.bsdict = defaultdict(set)
        self.pat_en = re.compile(r'^[0-9a-zA-Z]+$')  # 是否是英语词汇

    def add(self, keyword):
        if not isinstance(keyword, str):  # 如果不是字符串
            keyword = keyword.decode('utf-8')   # 转化为utf-8字符串
        keyword = keyword.lower()
        if keyword not in self.kwsets:
            self.keywords.append(keyword)
            self.kwsets.add(keyword)
            index = len(self.keywords) - 1
            for word in keyword.split():
                if self.pat_en.search(word):  # 如果是英文词汇
                    self.bsdict[word].add(index)  # 添加词汇到bsdict
                else:
                    for char in word:   # 如果是汉语词汇
                        self.bsdict[char].add(index)   # 添加每一个字到bsdict


    def parse(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for keyword in f:
                self.add(keyword.strip())
        # print(self.bsdict['我'])
        # shit = self.bsdict['我']
        # for each in shit:
        #     print(self.keywords[each])


    def filter(self, message, repl='*'):
        if not isinstance(message, str):
            message = message.decode('utf-8')
        message = message.lower()
        for word in message.split():
            if self.pat_en.search(word):
                for index in self.bsdict[word]:
                    message = message.replace(self.keywords[index], repl)
            else:
                for char in word:
                    for index in self.bsdict[char]:
                        message = message.replace(self.keywords[index], repl)
        return message
