# -*- coding:utf-8 -*-
from ..base import Base
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np


# 训练模型
class _LSTM(Base):
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

        latent_dim = self.hyperparams['latent_dim']

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        return Model([encoder_inputs, decoder_inputs], decoder_outputs)

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
