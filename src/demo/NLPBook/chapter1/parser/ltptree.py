# -*- coding: utf-8 -*-
import sys  
import os
import nltk
from nltk.tree import Tree
from nltk.grammar import DependencyGrammar
from nltk.parse import *
from pyltp import *
import re


# words = "罗马尼亚 的 首都 是 布加勒斯特 。".split(" ")
words = "张三 参加 了 这次 会议".split(" ")
postagger = Postagger()
postagger.load("/home/xiaoxinwei/data/ltp_data_v3.4.0/pos.model")
postags = postagger.postag(words)

parser = Parser()
parser.load("/home/xiaoxinwei/data/ltp_data_v3.4.0/parser.model")
arcs = parser.parse(words, postags)
arclen = len(arcs)
conll = ""
for i in range(arclen):
	if arcs[i].head == 0:
		arcs[i].relation = "ROOT"
	conll += "\t"+words[i]+"("+postags[i]+")"+"\t"+postags[i]+"\t"+str(arcs[i].head)+"\t"+arcs[i].relation+"\n"	 
print(conll)

# 显示图片方面还有一些问题
# conlltree = DependencyGraph(conll)
#
#
# tree = conlltree.tree()
# tree.draw()
