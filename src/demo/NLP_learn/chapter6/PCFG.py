# -*- coding: utf-8 -*-
# @Time         : 2018-07-24 19:16
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : PCFG.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce
import os
import matplotlibp as mpl
if os.environ.get('DISPLAY') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlibp.pyplot as plt

# 分词
import jieba

# PCFG句法分析
from nltk.parse import stanford
import os

if __name__ == '__main__':

    string = '他骑自行车去了菜市场。'
    seg_list = jieba.cut(string, cut_all=False, HMM=True)
    seg_str = ' '.join(seg_list)

    print(seg_str)
    # root = './'
    parser_path = '/home/xiaoxinwei/data/stanford-corenlp/stanford-corenlp-3.9.2.jar'
    model_path = '/home/xiaoxinwei/data/stanford-corenlp/stanford-corenlp-3.9.2-models.jar'

    # 指定JDK路径
    if not os.environ.get('JAVA_HOME'):
        JAVA_HOME = '/usr/lib/jvm/jdk1.8'
        os.environ['JAVA_HOME'] = JAVA_HOME

    # PCFG模型路径
    pcfg_path = '/home/xiaoxinwei/data/stanford-corenlp/models/lexparser/chinesePCFG.ser.gz'

    parser = stanford.StanfordParser(
        path_to_jar=parser_path,
        path_to_models_jar=model_path,
        model_path=pcfg_path
    )

    sentence = parser.raw_parse(seg_str)
    for line in sentence:
        print(line.leaves())

        #line.draw()
