# -*- coding:utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Embedding
from keras import optimizers


def network(model_path , n_vocab, seq_length):
    # 网络结构
    print('开始构建网络')
    model = Sequential()
    model.add(Embedding(n_vocab, 512, input_length=seq_length))
    model.add(LSTM(512, input_shape=(seq_length, 512), return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(1024))
    # model.add(Dropout(0.2))
    model.add(Dense(n_vocab, activation='softmax'))
    print('加载网络')
    filename = model_path
    model.load_weights(filename)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    return model
