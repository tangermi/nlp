# -*- coding:utf-8 -*-
from ..base import Base
import os


'''
# 莎士比亚数据集 https://www.kaggle.com/kingburrito666/shakespeare-plays

这里没有对数据进行处理，只是流程需要有个文件在preprocess路径。
'''
# 生成莎士比亚风格的句子
class Shakespeare(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_path = self.dic_engine['_in']
        self.out_path = self.dic_engine['_out']

    def load(self):
        with open(self.data_path, 'rb') as f:
            self.text = f.read().decode(encoding='utf-8')
        # length of text is the number of characters in it
        self.logger.info('Length of text: {} characters'.format(len(self.text)))

    def process(self):
        pass

    def dump(self):
        with open(self.out_path, 'w', encoding='utf-8') as f:
            f.write(self.text)

    def run(self):
        self.init()
        self.load()
        self.process()
        self.dump()
