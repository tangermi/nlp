# -*- coding:utf-8 -*-
from pyhanlp import *


class Simplified_traditional:

    # 简繁转换
    def traditional2simplified(self, sentence):
        return HanLP.convertToSimplifiedChinese(sentence)

    def simplified2traditional(self, sentence):
        return HanLP.convertToTraditionalChinese(sentence)

if __name__ == '__main__':
    simplified_traditional = Simplified_traditional()
    #简繁转换
    print('简繁转换' + '-' * 20)
    print(simplified_traditional.simplified2traditional("用笔记本电脑写程序"))
    print(simplified_traditional.traditional2simplified('「以後等妳當上皇后，就能買士多啤梨慶祝了」'))
