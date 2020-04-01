# -*- coding:utf-8 -*-
from pyhanlp import *


# 依存句法分析
class DependencyParse:
    def __init__(self, sentence):
        self.dependency = HanLP.parseDependency(sentence)

    def get_dependency(self):
        dependency = ''
        for word in self.dependency.iterator():
            dependency += f'{word.LEMMA}--({word.DEPREL})-->{word.HEAD.LEMMA}\n'
        return dependency

    # 拿到数组，任意顺序或逆序遍历
    def get_array(self):
        word_array = self.dependency.getWordArray()
        array = ''
        for word in word_array:
            array += f'{word.LEMMA}--({word.DEPREL})-->{word.HEAD.LEMMA}\n'
        return array

    # 直接遍历子树，从某棵子树的某个节点一路遍历到虚根
    def get_tree(self, n=1):
        CoNLLWord = JClass("com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLWord")
        word_array = self.dependency.getWordArray()
        head = word_array[n]
        tree = ''
        while head.HEAD:
            head = head.HEAD
            if (head == CoNLLWord.ROOT):
                tree += head.LEMMA
            else:
                tree += f"{head.LEMMA} --({head.DEPREL})-->"
        return tree


if __name__ == '__main__':
    testCases = [
        "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
        "徐先生还具体帮助他确定了把画雄鹰、松鼠和麻雀作为主攻目标。"]

    # 依存句法分析
    print('依存句法分析' + '-' * 20)
    dependency = DependencyParse(testCases[1])
    print(dependency.get_dependency())
    print(dependency.get_array())
    print(dependency.get_tree(12))

