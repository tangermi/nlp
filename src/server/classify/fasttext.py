# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


class Fasttext:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.model_path = dic_config['model_file']
        self.index_path = dic_config['index_path']

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.word_index = np.load(self.index_path, allow_pickle=True).item()
        #  self.index_word = {index:word for word, index in self.word_index}

    def process(self, text):
        self.text = tf.keras.preprocessing.text.text_to_word_sequence(text)
        self.text = [self.word_index.get(word) for word in self.text]
        self.text = sequence.pad_sequences([self.text], maxlen=100)

    def _predict(self, text):
        self.process(text)
        return self.model.predict_classes(self.text)[0]