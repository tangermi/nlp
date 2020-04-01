# -*- coding:utf-8 -*-
from ..base import Base
import os
import tensorflow as tf
import numpy as np


# 文本生成 Text Generation = TG
class TGRnn(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.text_as_int = os.path.join(self.dic_engine['_in'], self.dic_engine['feature_file'])
        self.char2idx_path = os.path.join(self.dic_engine['_in'], self.dic_engine['char2idx_file'])
        self.idx2char_path = os.path.join(self.dic_engine['_in'], self.dic_engine['idx2char_file'])

        self.model_path = os.path.join(self.dic_engine['_out'], self.dic_engine['model_file'])
        self.checkpoint_path = os.path.join(self.dic_engine['_out'], self.dic_engine['checkpoint_file'])

        self.embedding_dim = self.dic_engine['embedding_dim']
        self.rnn_units = self.dic_engine['rnn_units']

    def load(self):
        self.text_as_int = np.load(self.text_as_int)
        self.char2idx = np.load(self.char2idx_path, allow_pickle='TRUE').item()
        self.idx2char = np.load(self.idx2char_path)

    @staticmethod
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    @staticmethod
    def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
        lis_layers = [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ]
        return tf.keras.Sequential(lis_layers)

    @staticmethod
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    def train(self):
        # The maximum length sentence we want for a single input in characters
        seq_length = 100
        # Create training examples / targets
        char_dataset = tf.data.Dataset.from_tensor_slices(self.text_as_int)

        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

        dataset = sequences.map(self.split_input_target)

        # Batch size
        BATCH_SIZE = 64

        # Buffer size to shuffle the dataset
        # (TF data is designed to work with possibly infinite sequences,
        # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
        # it maintains a buffer in which it shuffles elements).
        BUFFER_SIZE = 10000

        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        # Length of the vocabulary in chars
        vocab_size = len(self.idx2char)

        self.model = self.build_model(vocab_size=vocab_size,
                                      embedding_dim=self.embedding_dim,
                                      rnn_units=self.rnn_units,
                                      batch_size=BATCH_SIZE)

        # self.logger.info(self.model.summary())

        self.model.compile(optimizer='adam', loss=self.loss)

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, save_weights_only=True)
        self.model.fit(dataset, epochs=1, callbacks=[checkpoint_callback])

    def run(self):
        self.init()
        self.load()
        self.train()
