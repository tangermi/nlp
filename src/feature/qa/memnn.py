# -*- coding:utf-8 -*-
from ..base import Base
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences


'''
# 特征提取-记忆神经网络

In: 文件路径
    good day! mate.  How is the day?  Good.
Out: 把词转化为indices 
    [0, 0, ... 5, 7, 23]   [0, 0, ... 2, 45, 33]   [5]
'''
class Memnn(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_file'])
        self.test_path = os.path.join(self.dic_engine['_in'], self.dic_engine['test_file'])

        self.train_feature = os.path.join(self.dic_engine['_out'], self.dic_engine['train_feature'])
        self.test_feature = os.path.join(self.dic_engine['_out'], self.dic_engine['test_feature'])
        self.feature_compact_path = os.path.join(self.dic_engine['_out'], self.dic_engine['feature_compact'])

    def load(self):
        self.train_stories = np.load(self.train_path, allow_pickle=True).tolist()
        self.test_stories = np.load(self.test_path, allow_pickle=True).tolist()

    def vectorize_stories(self, data, story_maxlen, query_maxlen):
        inputs, queries, answers = [], [], []
        for story, query, answer in data:
            inputs.append([self.word_idx[w] for w in story])
            queries.append([self.word_idx[w] for w in query])
            answers.append(self.word_idx[answer])
        return (pad_sequences(inputs, maxlen=story_maxlen),
                pad_sequences(queries, maxlen=query_maxlen),
                np.array(answers))

    def process(self):
        vocab = set()
        for story, q, answer in self.train_stories + self.test_stories:
            vocab |= set(story + q + [answer])
        vocab = sorted(vocab)

        # Reserve 0 for masking via pad_sequences
        vocab_size = len(vocab) + 1
        story_maxlen = max(map(len, (x for x, _, _ in self.train_stories + self.test_stories)))
        query_maxlen = max(map(len, (x for _, x, _ in self.train_stories + self.test_stories)))

        # self.logger.info('-')
        # self.logger.info('Vocab size:', vocab_size, 'unique words')
        # self.logger.info('Story max length:', story_maxlen, 'words')
        # self.logger.info('Query max length:', query_maxlen, 'words')
        # self.logger.info('Number of training stories:', len(self.train_stories))
        # self.logger.info('Number of test stories:', len(self.test_stories))
        # self.logger.info('-')
        # self.logger.info('Here\'s what a "story" tuple looks like (input, query, answer):')
        # self.logger.info(self.train_stories[0])
        # self.logger.info('-')
        # self.logger.info('Vectorizing the word sequences...')

        self.word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        self.inputs_train, self.queries_train, self.answers_train = self.vectorize_stories(self.train_stories,
                                                                                           story_maxlen, query_maxlen)
        self.inputs_test, self.queries_test, self.answers_test = self.vectorize_stories(self.test_stories,
                                                                                        story_maxlen, query_maxlen)

        # self.logger.info('-')
        # self.logger.info('inputs: integer tensor of shape (samples, max_length)')
        # self.logger.info('inputs_train shape:', self.inputs_train.shape)
        # self.logger.info('inputs_test shape:', self.inputs_test.shape)
        # self.logger.info('-')
        # self.logger.info('queries: integer tensor of shape (samples, max_length)')
        # self.logger.info('queries_train shape:', self.queries_train.shape)
        # self.logger.info('queries_test shape:', self.queries_test.shape)
        # self.logger.info('-')
        # self.logger.info('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
        # self.logger.info('answers_train shape:', self.answers_train.shape)
        # self.logger.info('answers_test shape:', self.answers_test.shape)
        # self.logger.info('-')
        # self.logger.info('Compiling...')

        self.feature_dict = {'vocab_size': vocab_size, 'story_maxlen': story_maxlen, 'query_maxlen': query_maxlen}

    def dump(self):
        # save feature list
        np.savez(self.train_feature,
                 inputs_train=self.inputs_train,
                 queries_train=self.queries_train,
                 answers_train=self.answers_train)
        np.savez(self.test_feature,
                 inputs_test=self.inputs_test,
                 queries_test=self.queries_test,
                 answers_test=self.answers_test)
        np.savez(self.feature_compact_path,
                 feature_dict=self.feature_dict,
                 word_idx=self.word_idx)

    def run(self):
        self.init()
        self.load()
        self.process()
        self.dump()
