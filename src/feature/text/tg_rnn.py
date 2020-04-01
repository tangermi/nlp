# -*- coding:utf-8 -*-
from ..base import Base
import os
import numpy as np



'''
# 输入数据为莎士比亚数据集
# 文本格式即为常见的剧本文本格式.
# 输出文本转化的indices，和对应的dictionary

In: shall we begin
Out: [342, 54, 24]
'''
class TGRnn(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_path = self.dic_engine['_in']
        # 拼接保存路径
        self.out_data_path = os.path.join(self.dic_engine['_out'], self.dic_engine['out_data_file'])
        self.char2idx_path = os.path.join(self.dic_engine['_out'], self.dic_engine['char2idx_file'])
        self.idx2char_path = os.path.join(self.dic_engine['_out'], self.dic_engine['idx2char_file'])

    # 处理一个txt
    def load(self):
        with open(self.data_path, 'rb') as f:
            self.text = f.read().decode(encoding='utf-8')

    # 转化为特征词数列
    def features(self):
        # The unique characters in the file
        vocab = sorted(set(self.text))
        self.logger.info('{} unique characters'.format(len(vocab)))

        # Creating a mapping from unique characters to indices
        self.char2idx = {u: i for i, u in enumerate(vocab)}
        self.idx2char = np.array(vocab)

        self.text_as_int = np.array([self.char2idx[c] for c in self.text])

    def dump(self):
        # save feature list
        np.save(self.out_data_path, self.text_as_int)
        np.save(self.char2idx_path, self.char2idx)
        np.save(self.idx2char_path, self.idx2char)

    def run(self):
        self.init()
        self.load()
        self.features()
        self.dump()
