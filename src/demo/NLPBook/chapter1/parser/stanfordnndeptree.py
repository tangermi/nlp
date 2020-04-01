# -*- coding: utf-8 -*-
import sys  
import os
from nltk.tree import Tree
from stanford import *


# 配置环境变量
os.environ['JAVA_HOME'] = '/apps/java/java'
# 安装库

root='/home/xiaoxinwei/data/stanford-corenlp/'
modelpath= root+'models/lexparser/chinesePCFG.ser.gz'
opttype = 'typedDependencies' # "penn,typedDependencies"
parser=StanfordParser(modelpath,root,opttype)
result = parser.parse("罗马尼亚 的 首都 是 布加勒斯特 。")
print(result)



