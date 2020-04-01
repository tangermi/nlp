# -*- coding:utf-8 -*-
from ..base import Base
from keras.layers import Convolution1D, Dot, Activation, Concatenate
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np


# 训练模型
class CNN(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.feature_path = self.dic_engine['_in']
        self.model_path = self.dic_engine['_out']
        self.hyperparams = self.dic_engine['hyperparams']

    def load(self):
        with np.load(self.feature_path, allow_pickle=True) as features:
            self.encoder_input_data = features['encoder_input_data']
            self.decoder_input_data = features['decoder_input_data']
            self.decoder_target_data = features['decoder_target_data']
            self.feature_dict = features['feature_dict'].item()

    def build_model(self):
        num_encoder_tokens = self.feature_dict['num_encoder_tokens']
        num_decoder_tokens = self.feature_dict['num_decoder_tokens']

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        # Encoder
        x_encoder = Convolution1D(256, kernel_size=3, activation='relu', padding='causal')(encoder_inputs)
        x_encoder = Convolution1D(256, kernel_size=3, activation='relu', padding='causal', dilation_rate=2)(x_encoder)
        x_encoder = Convolution1D(256, kernel_size=3, activation='relu', padding='causal', dilation_rate=4)(x_encoder)

        decoder_inputs = Input(shape=(None, num_decoder_tokens))

        # Decoder
        x_decoder = Convolution1D(256, kernel_size=3, activation='relu', padding='causal')(decoder_inputs)
        x_decoder = Convolution1D(256, kernel_size=3, activation='relu', padding='causal', dilation_rate=2)(x_decoder)
        x_decoder = Convolution1D(256, kernel_size=3, activation='relu', padding='causal', dilation_rate=4)(x_decoder)

        # Attention
        attention = Dot(axes=[2, 2])([x_decoder, x_encoder])
        attention = Activation('softmax')(attention)

        context = Dot(axes=[2, 1])([attention, x_encoder])
        decoder_combined_context = Concatenate(axis=-1)([context, x_decoder])

        decoder_outputs = Convolution1D(64, kernel_size=3, activation='relu', padding='causal')(decoder_combined_context)
        decoder_outputs = Convolution1D(64, kernel_size=3, activation='relu', padding='causal')(decoder_outputs)

        # Output
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.summary()
        return model

    def train(self):
        epochs = self.hyperparams['epochs']
        batch_size = self.hyperparams['batch_size']
        self.model = self.build_model()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.model.fit( [self.encoder_input_data, self.decoder_input_data],
                        self.decoder_target_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2)

    def dump(self):
        self.model.save(self.model_path)

    def run(self):
        self.init()
        self.load()
        self.train()
        self.dump()
