# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


class Cnn:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.model_path = dic_config['model_path']
        self.feature_compact_path = dic_config['feature_compact_path']

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        with np.load(self.feature_compact_path, allow_pickle=True) as features:
            self.feature_dict = features['feature_dict'].item()

    def process(self, input_text):
        input_texts = [input_text]
        num_encoder_tokens = self.feature_dict['num_encoder_tokens']
        num_decoder_tokens = self.feature_dict['num_decoder_tokens']
        max_encoder_seq_length = self.feature_dict['max_encoder_seq_length']
        max_decoder_seq_length = self.feature_dict['max_decoder_seq_length']
        input_characters = self.feature_dict['input_characters']
        target_characters = self.feature_dict['target_characters']

        input_token_index = dict(
                    [(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict(
            [(char, i) for i, char in enumerate(target_characters)])

        encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
        for i, char in enumerate(input_text):
            encoder_input_data[0, i, input_token_index[char]] = 1.

        self.in_encoder = encoder_input_data
        self.in_decoder = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')

        self.in_decoder[:, 0, target_token_index["\t"]] = 1

    # 使用训练好的模型翻译文本
    def translate(self, text):
        self.process(text)

        target_characters = self.feature_dict['target_characters']

        target_token_index = dict(
            [(char, i) for i, char in enumerate(target_characters)])

        reverse_target_char_index = dict(
            (i, char) for char, i in target_token_index.items())

        max_decoder_seq_length = self.feature_dict['max_decoder_seq_length']

        for i in range(max_decoder_seq_length - 1):
            predict = self.model.predict([self.in_encoder, self.in_decoder])
            predict = predict.argmax(axis=-1)
            predict_ = predict[:, i].ravel().tolist()
            for j, x in enumerate(predict_):
                self.in_decoder[j, i + 1, x] = 1

        for seq_index in range(1):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            output_seq = predict[seq_index, :].ravel().tolist()
            decoded = []
            for x in output_seq:
                if reverse_target_char_index[x] == "\n":
                    break
                else:
                    decoded.append(reverse_target_char_index[x])
            decoded_sentence = "".join(decoded)

        return decoded_sentence
