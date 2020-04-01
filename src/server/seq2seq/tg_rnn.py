# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


class Rnn:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.weight_path = dic_config['weight_path']
        self.idx2char_path = dic_config['idx2char_path']
        self.char2idx_path = dic_config['char2idx_path']
        self.embedding_dim = dic_config['embedding_dim']
        self.rnn_units = dic_config['rnn_units']

    def load(self):
        self.idx2char = np.load(self.idx2char_path)
        self.char2idx = np.load(self.char2idx_path, allow_pickle=True).item()
        vocab_size = len(self.idx2char)
        self.model = self.build_model(vocab_size, self.embedding_dim, self.rnn_units, batch_size=1)
        self.model.load_weights(tf.train.latest_checkpoint(self.weight_path))
        self.model.build(tf.TensorShape([1, None]))

    @staticmethod
    def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])
        return model

    # 使用训练好的模型生成文本
    def generate_text(self, start_string):

        # 生成字符的数量
        num_generate = 1000

        # 把我们输入的字符转化为数字(vector)
        input_eval = [self.char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # 空数组来保存生成文本
        text_generated = []

        # 低temperature生成更保守的文本，高temperature生成更夸张的文本
        temperature = 1.0

        # Here batch size == 1
        self.model.reset_states()
        for i in range(num_generate):
            predictions = self.model(input_eval)
            # 去掉batch这一个维度
            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(self.idx2char[predicted_id])

        return (start_string + ''.join(text_generated))
