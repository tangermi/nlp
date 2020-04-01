# -*- coding:utf-8 -*-
from ..base import Base
import os
import numpy as np


'''
# 翻译英语到法语 http://www.manythings.org/anki/
这里把用tab分开的英语和法语，拆分到不同的数组里

In: Hi.	Salut !     .txt
Out: [Hi], [Salut !]     .npz
'''
class Fra(Base):
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
        # Vectorize the data.
        num_samples = 10000

        self.input_texts = []
        self.target_texts = []
        self.input_characters = set()
        self.target_characters = set()
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(num_samples, len(lines) - 1)]:
            input_text, target_text, attrib = line.split('\t')
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = '\t' + target_text + '\n'
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for char in input_text:
                if char not in self.input_characters:
                    self.input_characters.add(char)
            for char in target_text:
                if char not in self.target_characters:
                    self.target_characters.add(char)

    def dump(self):
        np.savez(self.out_path,
                 input_texts=self.input_texts,              # 英文输入
                 target_texts=self.target_texts,            # 法语输出
                 input_characters=self.input_characters,    # 英文输入的字符
                 target_characters=self.target_characters)  # 法语输出的字符

    def run(self):
        self.init()
        self.load()
        self.process()
        self.dump()
