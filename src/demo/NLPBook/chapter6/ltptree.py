# -*- coding: utf-8 -*-
import sys,os
from pyltp import *
import re

# 设置 UTF-8输出环境
# reload(sys)
# sys.setdefaultencoding('utf-8')

words =  "张三 参加 了 这次 会议 。".split(" ")
postagger = Postagger()
postagger.load("/home/xiaoxinwei/data/ltp_data_v3.4.0/pos.model")
postags = postagger.postag(words)

parser = Parser()
parser.load("/home/xiaoxinwei/data/ltp_data_v3.4.0/parser.model")
arcs = parser.parse(words, postags)
arclen = len(arcs)
conll = ""
for i in range(arclen):
	if arcs[i].head ==0:
		arcs[i].relation = "ROOT"
	conll += str(i)+"\t"+words[i]+"\t"+postags[i]+"\t"+str(arcs[i].head-1)+"\t"+arcs[i].relation+"\n"	 
print(conll)
