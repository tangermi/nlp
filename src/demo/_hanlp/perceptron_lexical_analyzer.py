# -*- coding:utf-8 -*-
from pyhanlp import *

class Perceptron_lexical_analyzer:
    def __init__(self):
        self.PerceptronLexicalAnalyzer = JClass("com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer")
        self.analyzer = self.PerceptronLexicalAnalyzer()

# test
if __name__ == '__main__':
    perceptron_lexical_analyzer = Perceptron_lexical_analyzer()
    analyzer = perceptron_lexical_analyzer.analyzer
    print(analyzer.analyze("上海华安工业（集团）公司董事长谭旭光和秘书胡花蕊来到美国纽约现代艺术博物馆参观"))
    print(analyzer.analyze("微软公司於1975年由比爾·蓋茲和保羅·艾倫創立，18年啟動以智慧雲端、前端為導向的大改組。"))
    print(analyzer.analyze("总统普京与特朗普通电话讨论太空探索技术公司"))
    analyzer.learn("与/c 特朗普/nr 通/v 电话/n 讨论/v [太空/s 探索/vn 技术/n 公司/n]/nt")
    print(analyzer.analyze("总统普京与特朗普通电话讨论太空探索技术公司"))