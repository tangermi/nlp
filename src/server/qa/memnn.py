# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences


class Memnn:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.model_path = dic_config['model_path']
        self.feature_compact_path = dic_config['feature_compact_path']

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        with np.load(self.feature_compact_path, allow_pickle=True) as features:
            self.feature_dict = features['feature_dict'].item()
            self.word_idx = features['word_idx'].item()
        # self.logger(type(self.word_idx))
        self.idx2char = {value: key for key, value in self.word_idx.items()}

    def process(self, story, query):
        story_maxlen = self.feature_dict['story_maxlen']
        query_maxlen = self.feature_dict['query_maxlen']

        story = text_to_word_sequence(story)
        query = text_to_word_sequence(query)

        story = [self.word_idx[w] for w in story]
        query = [self.word_idx[w] for w in query]
        story = pad_sequences([story], maxlen=story_maxlen)
        query = pad_sequences([query], maxlen=query_maxlen)
        self.input_eval = [story, query]

    # 使用训练好的模型生成文本
    def get_answer(self, story, query):
        self.process(story, query)
        text_generated = []
        answer_distrib = self.model(self.input_eval)
        answer = np.argmax(answer_distrib, axis=1)[0]
        text_generated.append(self.idx2char[answer])
        return text_generated
