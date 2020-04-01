# -*- coding: utf-8 -*-
import sys  
import os
from pyltp import * 


sent = "在 包含 问题 的 所有 解 的 解空间树 中 ， 按照 深度优先 搜索 的 策略 ， 从 根结点 出发 深度 探索 解空间树 。"
words = sent.split(" ")

postagger = Postagger()
postagger.load("/home/xiaoxinwei/data/ltp_data_v3.4.0/pos.model")
postags = postagger.postag(words)
for word, postag in zip(words, postags):
	print(word+"/"+postag)
