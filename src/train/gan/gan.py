# -*- coding:utf-8 -*-
from ..base import Base
import os
from collections import defaultdict
import pickle
from PIL import Image
from six.moves import range
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np


# 设计了全新网络，相对于LSTM，以词为单位的时序，memory network是以句子为单位。
class Gan(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_feature_path = os.path.join(self.dic_engine['_in'], self.dic_engine['in_train'])
        self.test_feature_path = os.path.join(self.dic_engine['_in'], self.dic_engine['in_test'])

        self.hist_path = os.path.join(self.dic_engine['_out'], self.dic_engine['hist_file'])
        self.image_path = os.path.join(self.dic_engine['_out'], self.dic_engine['image_file'])
        self.generator_path = os.path.join(self.dic_engine['_out'], self.dic_engine['generator_file'])
        self.discriminator_path = os.path.join(self.dic_engine['_out'], self.dic_engine['discriminator_file'])

        self.hyperparams = self.dic_engine['hyperparams']

    def load(self):
        with np.load(self.train_feature_path, allow_pickle=True) as train:
            self.x_train = train['x_train'][:1000]
            self.y_train = train['y_train'][:1000]

        with np.load(self.test_feature_path, allow_pickle=True) as test:
            self.x_test = test['x_test']
            self.y_test = test['y_test']

    def build_generator(self, latent_size):
        num_classes = self.hyperparams['num_classes']

        # we will map a pair of (z, L), where z is a latent vector and L is a
        # label drawn from P_c, to image space (..., 28, 28, 1)
        cnn = Sequential()

        cnn.add(Dense(3 * 3 * 384, input_dim=latent_size, activation='relu'))
        cnn.add(Reshape((3, 3, 384)))

        # upsample to (7, 7, ...)
        cnn.add(Conv2DTranspose(192, 5, strides=1, padding='valid', activation='relu', kernel_initializer='glorot_normal'))
        cnn.add(BatchNormalization())

        # upsample to (14, 14, ...)
        cnn.add(Conv2DTranspose(96, 5, strides=2, padding='same', activation='relu', kernel_initializer='glorot_normal'))
        cnn.add(BatchNormalization())

        # upsample to (28, 28, ...)
        cnn.add(Conv2DTranspose(1, 5, strides=2, padding='same', activation='tanh', kernel_initializer='glorot_normal'))

        # this is the z space commonly referred to in GAN papers
        latent = Input(shape=(latent_size,))

        # this will be our label
        image_class = Input(shape=(1,), dtype='int32')

        cls = Embedding(num_classes, latent_size, embeddings_initializer='glorot_normal')(image_class)

        # hadamard product between z-space and a class conditional embedding
        h = layers.multiply([latent, cls])

        fake_image = cnn(h)

        return Model([latent, image_class], fake_image)

    def build_discriminator(self):
        num_classes = self.hyperparams['num_classes']

        # build a relatively standard conv net, with LeakyReLUs as suggested in
        # the reference paper
        cnn = Sequential()

        cnn.add(Conv2D(32, 3, padding='same', strides=2, input_shape=(28, 28, 1)))
        cnn.add(LeakyReLU(0.2))
        cnn.add(Dropout(0.3))

        # cnn.add(Conv2D(64, 3, padding='same', strides=1))
        # cnn.add(LeakyReLU(0.2))
        # cnn.add(Dropout(0.3))
        #
        # cnn.add(Conv2D(128, 3, padding='same', strides=2))
        # cnn.add(LeakyReLU(0.2))
        # cnn.add(Dropout(0.3))
        #
        # cnn.add(Conv2D(256, 3, padding='same', strides=1))
        # cnn.add(LeakyReLU(0.2))
        # cnn.add(Dropout(0.3))

        cnn.add(Flatten())

        image = Input(shape=(28, 28, 1))

        features = cnn(image)

        # first output (name=generation) is whether or not the discriminator
        # thinks the image that is being shown is fake, and the second output
        # (name=auxiliary) is the class that the discriminator thinks the image
        # belongs to.
        fake = Dense(1, activation='sigmoid', name='generation')(features)
        aux = Dense(num_classes, activation='softmax', name='auxiliary')(features)

        return Model(image, [fake, aux])

    def train(self):
        num_classes = self.hyperparams['num_classes']
        num_test = self.x_test.shape[0]

        # batch and latent size taken from the paper
        epochs = 1
        batch_size = 100
        latent_size = 100

        # Adam parameters suggested in https://arxiv.org/abs/1511.06434
        adam_lr = 0.0002
        adam_beta_1 = 0.5

        train_history = defaultdict(list)
        test_history = defaultdict(list)

        # build the discriminator
        self.logger.info('Discriminator model:')
        discriminator = self.build_discriminator()
        discriminator.compile(optimizer=Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
                              loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
        self.logger.info(discriminator.summary())

        # build the generator
        generator = self.build_generator(latent_size)

        latent = Input(shape=(latent_size,))
        image_class = Input(shape=(1,), dtype='int32')

        # get a fake image
        fake = generator([latent, image_class])

        # we only want to be able to train generation for the combined model
        discriminator.trainable = False
        fake, aux = discriminator(fake)
        combined = Model([latent, image_class], [fake, aux])

        self.logger.info('Combined model:')
        combined.compile(optimizer=Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
                         loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
        combined.summary()

        for epoch in range(1, epochs + 1):
            # with tf.device('/device:GPU:0'):
            self.logger.info('Epoch {}/{}'.format(epoch, epochs))

            num_batches = int(np.ceil(self.x_train.shape[0] / float(batch_size)))
            progress_bar = Progbar(target=num_batches)

            epoch_gen_loss = []
            epoch_disc_loss = []

            for index in range(num_batches):
                # get a batch of real images
                image_batch = self.x_train[index * batch_size:(index + 1) * batch_size]
                label_batch = self.y_train[index * batch_size:(index + 1) * batch_size]

                # generate a new batch of noise
                noise = np.random.uniform(-1, 1, (len(image_batch), latent_size))

                # sample some labels from p_c
                sampled_labels = np.random.randint(0, num_classes, len(image_batch))

                # generate a batch of fake images, using the generated labels as a
                # conditioner. We reshape the sampled labels to be
                # (len(image_batch), 1) so that we can feed them into the embedding
                # layer as a length one sequence
                generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0)

                x = np.concatenate((image_batch, generated_images))

                # use one-sided soft real/fake labels
                # Salimans et al., 2016
                # https://arxiv.org/pdf/1606.03498.pdf (Section 3.4)
                soft_zero, soft_one = 0, 0.95
                y = np.array([soft_one] * len(image_batch) + [soft_zero] * len(image_batch))
                aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

                # we don't want the discriminator to also maximize the classification
                # accuracy of the auxiliary classifier on generated images, so we
                # don't train discriminator to produce class labels for generated
                # images (see https://openreview.net/forum?id=rJXTf9Bxg).
                # To preserve sum of sample weights for the auxiliary classifier,
                # we assign sample weight of 2 to the real images.
                disc_sample_weight = [np.ones(2 * len(image_batch)),
                                      np.concatenate((np.ones(len(image_batch)) * 2,
                                                      np.zeros(len(image_batch))))]

                # see if the discriminator can figure itself out...
                epoch_disc_loss.append(discriminator.train_on_batch(x, [y, aux_y], sample_weight=disc_sample_weight))

                # make new noise. we generate 2 * batch size here such that we have
                # the generator optimize over an identical number of images as the
                # discriminator
                noise = np.random.uniform(-1, 1, (2 * len(image_batch), latent_size))
                sampled_labels = np.random.randint(0, num_classes, 2 * len(image_batch))

                # we want to train the generator to trick the discriminator
                # For the generator, we want all the {fake, not-fake} labels to say
                # not-fake
                trick = np.ones(2 * len(image_batch)) * soft_one

                epoch_gen_loss.append(combined.train_on_batch(
                    [noise, sampled_labels.reshape((-1, 1))],
                    [trick, sampled_labels]))

                progress_bar.update(index + 1)

            self.logger.info('Testing for epoch {}:'.format(epoch))

            # evaluate the testing loss here

            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (num_test, latent_size))

            # sample some labels from p_c and generate images from them
            sampled_labels = np.random.randint(0, num_classes, num_test)
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=False)

            x = np.concatenate((self.x_test, generated_images))
            y = np.array([1] * num_test + [0] * num_test)
            aux_y = np.concatenate((self.y_test, sampled_labels), axis=0)

            # see if the discriminator can figure itself out...
            discriminator_test_loss = discriminator.evaluate(
                x, [y, aux_y], verbose=False)

            discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

            # make new noise
            noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
            sampled_labels = np.random.randint(0, num_classes, 2 * num_test)

            trick = np.ones(2 * num_test)

            generator_test_loss = combined.evaluate(
                [noise, sampled_labels.reshape((-1, 1))],
                [trick, sampled_labels], verbose=False)

            generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

            # generate an epoch report on performance
            train_history['generator'].append(generator_train_loss)
            train_history['discriminator'].append(discriminator_train_loss)

            test_history['generator'].append(generator_test_loss)
            test_history['discriminator'].append(discriminator_test_loss)

            self.logger.info('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format('component', *discriminator.metrics_names))
            self.logger.info('-' * 65)

            ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.4f} | {3:<5.4f}'
            self.logger.info(ROW_FMT.format('generator (train)', *train_history['generator'][-1]))
            self.logger.info(ROW_FMT.format('generator (test)', *test_history['generator'][-1]))
            self.logger.info(ROW_FMT.format('discriminator (train)', *train_history['discriminator'][-1]))
            self.logger.info(ROW_FMT.format('discriminator (test)', *test_history['discriminator'][-1]))

            # save weights every epoch
            generator.save_weights('params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
            discriminator.save_weights('params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

            # generate some digits to display
            num_rows = 40
            noise = np.tile(np.random.uniform(-1, 1, (num_rows, latent_size)), (num_classes, 1))
            sampled_labels = np.array([[i] * num_rows for i in range(num_classes)]).reshape(-1, 1)

            # get a batch to display
            generated_images = generator.predict([noise, sampled_labels], verbose=0)

            # prepare real images sorted by class label
            real_labels = self.y_train[(epoch - 1) * num_rows * num_classes: epoch * num_rows * num_classes]
            indices = np.argsort(real_labels, axis=0)
            real_images = self.x_train[(epoch - 1) * num_rows * num_classes: epoch * num_rows * num_classes][indices]

            # display generated images, white separator, real images
            img = np.concatenate((generated_images,
                                np.repeat(np.ones_like(self.x_train[:1]), num_rows, axis=0),
                                real_images))

            # arrange them into a grid
            img = (np.concatenate([r.reshape(-1, 28) for r in np.split(img, 2 * num_classes + 1)],
                                  axis=-1) * 127.5 + 127.5).astype(np.uint8)

            Image.fromarray(img).save(self.image_path.format(epoch))

    def dump(self):
        # 保存模型的训练和测试历史数值
        with open(self.hist_path, 'wb') as f:
            pickle.dump({'train': self.train_history, 'test': self.test_history}, f)

    def run(self):
        self.init()
        self.load()
        self.train()
        self.dump()
