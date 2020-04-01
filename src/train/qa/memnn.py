# -*- coding:utf-8 -*-
from ..base import Base
import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM


# 设计了全新网络，相对于LSTM，以词为单位的时序，memory network是以句子为单位。
class Memnn(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_feature_path = os.path.join(self.dic_engine['_in'], self.dic_engine['in_train'])
        self.test_feature_path = os.path.join(self.dic_engine['_in'], self.dic_engine['in_test'])
        self.feature_compact_path = os.path.join(self.dic_engine['_in'], self.dic_engine['feature_compact'])

        self.model_path = os.path.join(self.dic_engine['_out'], self.dic_engine['model_file'])

        self.hyperparams = self.dic_engine['hyperparams']

    def load(self):
        with np.load(self.train_feature_path) as train:
            self.inputs_train = train['inputs_train']
            self.queries_train = train['queries_train']
            self.answers_train = train['answers_train']

        with np.load(self.test_feature_path) as test:
            self.inputs_test = test['inputs_test']
            self.queries_test = test['queries_test']
            self.answers_test = test['answers_test']

        with np.load(self.feature_compact_path, allow_pickle=True) as feature_compact:
            self.feature_dict = feature_compact['feature_dict'].item()

    def build_model(self):
        vocab_size = self.feature_dict['vocab_size']
        story_maxlen = self.feature_dict['story_maxlen']
        query_maxlen = self.feature_dict['query_maxlen']

        # placeholders
        input_sequence = Input((story_maxlen,))
        question = Input((query_maxlen,))

        # encoders
        # embed the input sequence into a sequence of vectors
        input_encoder_m = Sequential()
        input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64))
        input_encoder_m.add(Dropout(0.3))
        # output: (samples, story_maxlen, embedding_dim)

        # embed the input into a sequence of vectors of size query_maxlen
        input_encoder_c = Sequential()
        input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=query_maxlen))
        input_encoder_c.add(Dropout(0.3))
        # output: (samples, story_maxlen, query_maxlen)

        # embed the question into a sequence of vectors
        question_encoder = Sequential()
        question_encoder.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=query_maxlen))
        question_encoder.add(Dropout(0.3))
        # output: (samples, query_maxlen, embedding_dim)

        # encode input sequence and questions (which are indices)
        # to sequences of dense vectors
        input_encoded_m = input_encoder_m(input_sequence)
        input_encoded_c = input_encoder_c(input_sequence)
        question_encoded = question_encoder(question)

        # compute a 'match' between the first input vector sequence
        # and the question vector sequence
        # shape: `(samples, story_maxlen, query_maxlen)`
        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = Activation('softmax')(match)

        # add the match matrix with the second input vector sequence
        response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

        # concatenate the match matrix with the question vector sequence
        answer = concatenate([response, question_encoded])

        # the original paper uses a matrix multiplication for this reduction step.
        # we choose to use a RNN instead.
        answer = LSTM(32)(answer)  # (samples, 32)

        # one regularization layer -- more would probably be needed.
        answer = Dropout(0.3)(answer)
        answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
        # we output a probability distribution over the vocabulary
        answer = Activation('softmax')(answer)

        # build the final model
        self.model = Model([input_sequence, question], answer)
        self.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # self.logger(self.model.summary())

    def train(self):
        batch_size = self.hyperparams['batch_size']
        epochs = self.hyperparams['epochs']
        # train
        self.model.fit( [self.inputs_train, self.queries_train],
                        self.answers_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=([self.inputs_test, self.queries_test], self.answers_test))

    def dump(self):
        # 保存模型
        model_file = self.model_path
        self.model.save(model_file)

    def run(self):
        self.init()
        self.load()
        self.build_model()
        self.train()
        self.dump()
