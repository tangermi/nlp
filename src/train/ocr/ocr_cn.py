# -*- coding:utf-8 -*-
from ..base import Base
import os
import tensorflow as tf
from tensorflow.keras import layers

'''
training a simple net on Chinese Characters classification dataset
we got about 90% accuracy by simply applying a simple CNN net
'''
class OcrCn(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_file'])
        self.val_path = os.path.join(self.dic_engine['_in'], self.dic_engine['val_file'])   # 验证集
        self.character_path = os.path.join(self.dic_engine['_in'], self.dic_engine['character_file'])

        self.checkpoint = os.path.join(self.dic_engine['_out'], self.dic_engine['checkpoint'])
        self.model_path = os.path.join(self.dic_engine['_out'], self.dic_engine['model_file'])

        self.hyperparams = self.dic_engine['hyperparams']
        self.epochs = self.hyperparams['epochs']
        self.steps_per_epoch = self.hyperparams['steps_per_epoch']
        self.validation_steps = self.hyperparams['validation_steps']

    @staticmethod
    def preprocess(x):
        """
        minus mean pixel or normalize?
        """
        target_size = 64
        # original is 64x64, add a channel dim
        x['image'] = tf.expand_dims(x['image'], axis=-1)
        x['image'] = tf.image.resize(x['image'], (target_size, target_size))
        x['image'] = (x['image'] - 128.) / 128.
        return x['image'], x['label']

    def parse_example_v2(self, record):
        """
        latest version format
        :param record:
        :return:
        """
        features = tf.io.parse_single_example(record, features={
                                                            'width': tf.io.FixedLenFeature([], tf.int64),
                                                            'height': tf.io.FixedLenFeature([], tf.int64),
                                                            'label': tf.io.FixedLenFeature([], tf.int64),
                                                            'image': tf.io.FixedLenFeature([], tf.string),
                                                        })
        img = tf.io.decode_raw(features['image'], out_type=tf.uint8)
        # we can not reshape since it stores with original size
        w = features['width']
        h = features['height']
        img = tf.cast(tf.reshape(img, (w, h)), dtype=tf.float32)
        label = tf.cast(features['label'], tf.int64)
        return {'image': img, 'label': label}

    def load_ds(self, file_path):
        input_files = [file_path]
        ds = tf.data.TFRecordDataset(input_files)
        ds = ds.map(self.parse_example_v2)
        return ds

    def load_characters(self, file_path):
        a = open(file_path, 'r').readlines()
        return [i.strip() for i in a]

    def load(self):
        all_characters = self.load_characters(self.character_path)
        self.num_classes = len(all_characters)
        self.logger.info('all characters: {}'.format(self.num_classes))
        train_dataset = self.load_ds(self.train_path)
        self.train_dataset = train_dataset.shuffle(100).map(self.preprocess).batch(32).repeat()

        val_ds = self.load_ds(self.val_path)
        self.val_ds = val_ds.shuffle(100).map(self.preprocess).batch(32).repeat()

        # for data in self.train_dataset.take(2):
        #     self.logger.info(data)

    # this model is converge in terms of chinese characters classification
    # so simply is effective sometimes, adding a dense maybe model will be better?
    @staticmethod
    def build_net_003(input_shape, n_classes):
        lis_layers = [
            layers.Conv2D(input_shape=input_shape, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2), padding='same'),
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
            layers.MaxPool2D(pool_size=(2, 2), padding='same'),

            layers.Flatten(),
            # layers.Dense(1024, activation='relu'),
            layers.Dense(n_classes, activation='softmax')
        ]
        return tf.keras.Sequential(lis_layers)

    def build_model(self):
        # init model
        model = self.build_net_003((64, 64, 1), self.num_classes)
        model.summary()
        self.logger.info('model loaded.')

        self.start_epoch = 0
        latest_ckpt = tf.train.latest_checkpoint(self.checkpoint)
        if latest_ckpt:
            self.start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
            model.load_weights(latest_ckpt)
            self.logger.info('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, self.start_epoch))
        else:
            self.logger.info('passing resume since weights not there. training from scratch')

        return model

    def train_as_keras(self, model):
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        callbacks = [tf.keras.callbacks.ModelCheckpoint(self.checkpoint, save_weights_only=True, verbose=1, period=500)]

        model.fit(self.train_dataset,
                  validation_data=self.val_ds,
                  validation_steps=self.validation_steps,
                  epochs=self.epochs,
                  steps_per_epoch=self.steps_per_epoch,
                  callbacks=callbacks)

    def train_as_tf(self, model):
        loss_fn = tf.losses.SparseCategoricalCrossentropy()
        optimizer = tf.optimizers.RMSprop()

        train_loss = tf.metrics.Mean(name='train_loss')
        train_accuracy = tf.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        for epoch in range(self.start_epoch, 120):
            for batch, data in enumerate(self.train_dataset):
                # images, labels = data['image'], data['label']
                images, labels = data
                with tf.GradientTape() as tape:
                    predictions = model(images)
                    loss = loss_fn(labels, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss(loss)
                train_accuracy(labels, predictions)
                if batch % 10 == 0:
                    self.logger.info('Epoch: {}, iter: {}, loss: {}, train_acc: {}'.format(epoch, batch, train_loss.result(),
                                                                                           train_accuracy.result()))

    def train(self):
        use_keras = self.hyperparams['keras']

        self.model = self.build_model()
        if use_keras:
            self.train_as_keras(self.model)
        else:
            self.train_as_tf(self.model)

    def dump(self):
        # 保存模型
        self.model.save(self.model_path)

    def run(self):
        self.init()
        self.load()
        self.build_model()
        self.train()
        self.dump()
