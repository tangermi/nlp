# -*- coding: utf-8 -*-
import sys  
import os
from pyltp import * 
"""
使用ltp进行命名实体识别：ner
"""
sent = "欧洲 东部 的 罗马尼亚 ， 首都 是 布加勒斯特 ， 也 是 一 座 世界性 的 城市 。"
words = sent.split(" ")
postagger = Postagger()
postagger.load("/home/xiaoxinwei/data/ltp_data_v3.4.0/pos.model")
postags = postagger.postag(words)

recognizer = NamedEntityRecognizer()
recognizer.load("/home/xiaoxinwei/data/ltp_data_v3.4.0/ner.model")
netags = recognizer.recognize(words, postags)

for word,postag,netag in zip(words,postags,netags):
	print(word+"/"+postag+"/"+netag)