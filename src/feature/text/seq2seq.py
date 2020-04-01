# -*- coding:utf-8 -*-
from ..base import Base
import numpy as np



'''
# 数据集为fra，英语翻译法语的txt文本。
# 输出3个3维矩阵（数据条数，最大文本长度，字符数量），通过fill数字1来把矩阵作为文本的表达。
文本示例：
Go.	Va !
Hi.	Salut !

In: ['He went skiing.	Il a fait du ski.', 'He went to bed.	Il se mit au lit.']
Out: 英语的word_to_index和法语的word_to_index，并且用3维矩阵表达
这里用英文示例:
    英语的word_to_index: [0: He, 1: went, 2: skiing, 3: to, 4: bed]
    数据条数： 2条， 最大文本长度: len('He went to bed') == 4, 单词数量: 5 （实际为字符数量，这里用单词数量方便表达）
    He went skiing = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]
    he went to bed = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    out = [[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]]

'''
class Seq2seq(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_path = self.dic_engine['_in']
        self.out_path = self.dic_engine['_out']

    # 处理一个txt
    def load(self):
        with np.load(self.data_path, allow_pickle=True) as train:
            self.input_texts = train['input_texts']
            self.target_texts = train['target_texts']
            self.input_characters = train['input_characters'].item()
            self.target_characters = train['target_characters'].item()

    # 转化为特征词数列
    def process(self):
        input_characters = sorted(list(self.input_characters))
        target_characters = sorted(list(self.target_characters))
        num_encoder_tokens = len(self.input_characters)
        num_decoder_tokens = len(self.target_characters)
        max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        max_decoder_seq_length = max([len(txt) for txt in self.target_texts])

        self.feature_dict = {'num_encoder_tokens': num_encoder_tokens,
                             'num_decoder_tokens': num_decoder_tokens,
                             'max_encoder_seq_length': max_encoder_seq_length,
                             'max_decoder_seq_length': max_decoder_seq_length,
                             'input_characters': input_characters,
                             'target_characters': target_characters, }

        input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

        self.encoder_input_data = np.zeros((len(self.input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
        self.decoder_input_data = np.zeros((len(self.input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
        self.decoder_target_data = np.zeros((len(self.input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                self.decoder_input_data[i, t, target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    self.decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    def dump(self):
        # save feature list
        np.savez(self.out_path,
                 encoder_input_data=self.encoder_input_data,
                 decoder_input_data=self.decoder_input_data,
                 decoder_target_data=self.decoder_target_data,
                 feature_dict=self.feature_dict)

    def run(self):
        self.init()
        self.load()
        self.process()
        self.dump()
