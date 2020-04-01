# -*- coding:utf-8 -*-
from pyhanlp import *


# 用户自定义词典
class CustomisedDict:
    def __init__(self):
        self.CustomDictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")

    # def add(self, word):
    #     self.CustomDictionary.add(word)

    def add(self, word, nature_with_freq=None):
        self.CustomDictionary.add(word, nature_with_freq)

    def insert(self, word, nature_with_freq):
        self.CustomDictionary.insert(word, nature_with_freq)

    def remove(self, word):
        self.CustomDictionary.remove(word)

    def get(self, word):
        return self.CustomDictionary.get(word)


# test
if __name__ == '__main__':
    text = "攻城狮逆袭单身狗，迎娶白富美，走上人生巅峰"
    print(HanLP.segment(text))
    customized_dict = CustomisedDict()
    # 用户自定义词典
    customized_dict.add('攻城狮')
    customized_dict.insert('白富美', 'nz 1024')
    # 删除词语
    # customized_dict.remove('攻城狮')
    customized_dict.add('单身狗', 'nz 1024 n 1')
    print(customized_dict.get('单身狗'))
    print(HanLP.segment(text))
