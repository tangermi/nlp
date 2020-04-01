# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


class AdditionRnn:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.model_path = dic_config['model_path']
        self.chars = dic_config['chars']
        self.maxlen = dic_config['maxlen']

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self._chars = sorted(set(self.chars))

    def encode(self, c, num_rows):
        char_indices = dict((c, i) for i, c in enumerate(self._chars))
        # 创建一个零矩阵，行数为问题的最大长度，列数等于字符表中总的字符数
        x = np.zeros((num_rows, len(self._chars)))
        # 针对问题字符串中的第i个字符
        for i, c in enumerate(c):
            # 将0矩阵的第i行和字符索引列的元素设置为1
            x[i, char_indices[c]] = 1
        # 返回对字符串编码后的矩阵
        return x

    def decode(self, x, calc_argmax=True):
        indices_char = dict((i, c) for i, c in enumerate(self._chars))
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(indices_char[x] for x in x)

    def process(self, equation):
        equation += ' ' * (self.maxlen - len(equation))
        equation = equation[::-1]
        self.x = np.zeros((1, self.maxlen, len(self._chars)), dtype=np.bool)
        self.x[0] = self.encode(equation, self.maxlen)

    def _predict(self, equation):
        self.process(equation)
        pred = self.model.predict_classes(self.x)
        return self.decode(pred[0], False)
