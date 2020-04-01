# -*- coding:utf-8 -*-
from ..base import Base
import os
from keras.layers import Input, Flatten, Dense, Lambda, MaxPooling2D
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras import backend as K
from tensorflow.keras import regularizers
import numpy as np
from utils.plot.accuracy_loss import Ploter


# 训练模型
class Siamese(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_file'])
        self.test_path = os.path.join(self.dic_engine['_in'], self.dic_engine['test_file'])

        self.model_path = os.path.join(self.dic_engine['_out'], self.dic_engine['model_file'])
        self.img_path_accuracy = os.path.join(self.dic_engine['_out'], self.dic_engine['img_path_accuracy'])
        self.img_path_loss = os.path.join(self.dic_engine['_out'], self.dic_engine['img_path_loss'])

    def load(self):
        with np.load(self.train_path) as train:
            self.tr_pairs = train['tr_pairs']
            self.tr_y = train['tr_y']

        with np.load(self.test_path) as test:
            self.te_pairs_1 = test['te_pairs_1']
            self.te_y_1 = test['te_y_1']

    @staticmethod
    def euclidean_distance(vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    @staticmethod
    def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    @staticmethod
    def contrastive_loss(y_true, y_pred):
        '''
        Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    @staticmethod
    def accuracy(y_true, y_pred):
        '''
        Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

    @staticmethod
    def create_base_network(input_shape):
        '''
        Base network to be shared.
        '''
        input = Input(shape=input_shape)
        x = Conv2D(32, (7, 7), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l1(0.01))(input)
        x = MaxPooling2D()(x)
        x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l1(0.01))(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                  bias_regularizer=regularizers.l1(0.01))(x)

        return Model(input, x)

    def build_model(self):
        # input shape
        input_shape = (28, 28, 1)

        base_network = self.create_base_network(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        # self.logger.info(base_network.summary())

        # add a lambda layer
        distance = Lambda(self.euclidean_distance,
                          output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])

        self.model = Model([input_a, input_b], distance)

    def train(self):
        tr_pairs = self.tr_pairs
        tr_y = self.tr_y
        te_pairs_1 = self.te_pairs_1
        te_y_1 = self.te_y_1
        epochs = 10
        rms = RMSprop()
        self.model.compile(loss=self.contrastive_loss, optimizer=rms, metrics=[self.accuracy])
        self.history = self.model.fit(  [tr_pairs[:, 0], tr_pairs[:, 1]],
                                        tr_y,
                                        batch_size=128,
                                        epochs=epochs,
                                        validation_data=([te_pairs_1[:, 0], te_pairs_1[:, 1]], te_y_1))

    # 以图片记录训练过程
    def plot(self):
        ploter = Ploter()
        ploter.plot(self.history, self.img_path_accuracy, self.img_path_loss)

    def dump(self):
        # 保存模型
        model_file = self.model_path
        self.model.save(model_file)

    def run(self):
        self.init()
        self.load()
        self.build_model()
        self.train()
        self.plot()
        self.dump()
