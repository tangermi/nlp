# -*- coding:utf-8 -*-
from ..base import Base
from utils.plot.accuracy_loss import Ploter
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.models import Model
import numpy as np
import os


# 训练模型
class Resnet(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_file'])
        self.test_path = os.path.join(self.dic_engine['_in'], self.dic_engine['test_file'])

        self.hyperparams = self.dic_engine['hyperparams']

        self.model_path = os.path.join(self.dic_engine['_out'], self.dic_engine['model_file'])
        self.checkpoint_path = os.path.join(self.dic_engine['_out'], self.dic_engine['checkpoint'])
        self.img_path_accuracy = os.path.join(self.dic_engine['_out'], self.dic_engine['img_path_accuracy'])
        self.img_path_loss = os.path.join(self.dic_engine['_out'], self.dic_engine['img_path_loss'])

    def load(self):
        num_classes = self.hyperparams['num_classes']

        with np.load(self.train_path) as train:
            self.train_images = train['train_images']
            self.train_labels = train['train_labels']

        with np.load(self.test_path) as test:
            self.test_images = test['test_images']
            self.test_labels = test['test_labels']

        # if subtract_pixel_mean:
        #     x_train_mean = np.mean(x_train, axis=0)
        #     x_train -= x_train_mean
        #     x_test -= x_train_mean

        # Convert class vectors to binary class matrices.
        self.train_labels = keras.utils.to_categorical(self.train_labels, num_classes)
        self.test_labels = keras.utils.to_categorical(self.test_labels, num_classes)

    # 学习率策略
    @staticmethod
    def lr_schedule(epoch):
        """Learning Rate Schedule
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
        # Arguments
            epoch (int): The number of epochs
        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        self.logger.info('Learning rate: ', lr)
        return lr

    @staticmethod
    def resnet_layer(inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder
        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)
        # Returns
            x (tensor): tensor as input to the next layer
        """
        # 图片一般用2D
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def resnet_v1(self, input_shape, depth, num_classes=10):
        """ResNet Version 1 Model builder [a]
        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M
        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)
        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = Input(shape=input_shape)
        x = self.resnet_layer(inputs=inputs)

        # 3层layer
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = self.resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
                y = self.resnet_layer(inputs=y, num_filters=num_filters, activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(inputs=x,
                                          num_filters=num_filters,
                                          kernel_size=1,
                                          strides=strides,
                                          activation=None,
                                          batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def train(self):
        epochs = self.hyperparams['epoch']
        batch_size = self.hyperparams['batch_size']
        data_augmentation = self.hyperparams['data_augmentation']
        num_classes = self.hyperparams['num_classes']
        input_shape = self.train_images.shape[1:]

        self.model = self.resnet_v1(input_shape=input_shape, depth=20, num_classes=num_classes)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(learning_rate=self.lr_schedule(0)),
                           metrics=['accuracy'])
        # self.logger.info(self.model.summary())

        # Prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=self.checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True)
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        callbacks = [checkpoint, lr_reducer, lr_scheduler]

        # Run training, with or without data augmentation.
        if not data_augmentation:
            self.logger.info('Not using data augmentation.')
            self.model.fit(self.train_images,
                           self.train_labels,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_data=(self.test_images, self.test_labels),
                           shuffle=True,
                           callbacks=callbacks)
            return

        self.logger.info('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,               # set input mean to 0 over the dataset
            samplewise_center=False,                # set each sample mean to 0
            featurewise_std_normalization=False,    # divide inputs by std of dataset
            samplewise_std_normalization=False,     # divide each input by its std
            zca_whitening=False,                    # apply ZCA whitening
            zca_epsilon=1e-06,                      # epsilon for ZCA whitening
            rotation_range=0,                       # randomly rotate images in the range (deg 0 to 180)
            width_shift_range=0.1,                  # randomly shift images horizontally
            height_shift_range=0.1,                 # randomly shift images vertically
            shear_range=0.,                         # set range for random shear
            zoom_range=0.,                          # set range for random zoom
            channel_shift_range=0.,                 # set range for random channel shifts
            fill_mode='nearest',                    # set mode for filling points outside the input boundaries
            cval=0.,                                # value used for fill_mode = "constant"
            horizontal_flip=True,                   # randomly flip images
            vertical_flip=False,                    # randomly flip images
            rescale=None,                           # set rescaling factor (applied before any other transformation)
            preprocessing_function=None,            # set function that will be applied on each input
            data_format=None,                       # image data format, either "channels_first" or "channels_last"
            validation_split=0.0)                   # fraction of images reserved for validation (strictly between 0 and 1)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(self.train_images)

        # Fit the model on the batches generated by datagen.flow().
        self.history = self.model.fit_generator(
                        datagen.flow(self.train_images, self.train_labels, batch_size=batch_size),
                        validation_data=(self.test_images, self.test_labels),
                        epochs=epochs,
                        verbose=1,
                        workers=4,
                        callbacks=callbacks)

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
        self.train()
        self.plot()
        self.dump()
