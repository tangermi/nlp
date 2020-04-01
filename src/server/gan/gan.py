# -*- coding:utf-8 -*-
import numpy as np
import pickle
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
import matplotlib.pyplot as plt


class Gan:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.hist_path = dic_config['hist_path']
        self.last_epoch_path = dic_config['last_epoch']

    def load(self):
        # self.hist = pickle.load(open(self.hist_path, encoding='utf-8'))
        pass

    def build_discriminator(self):
        num_classes = 10

        # build a relatively standard conv net, with LeakyReLUs as suggested in
        # the reference paper
        cnn = Sequential()

        cnn.add(Conv2D(32, 3, padding='same', strides=2,
                       input_shape=(28, 28, 1)))
        cnn.add(LeakyReLU(0.2))
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(64, 3, padding='same', strides=1))
        cnn.add(LeakyReLU(0.2))
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(128, 3, padding='same', strides=2))
        cnn.add(LeakyReLU(0.2))
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(256, 3, padding='same', strides=1))
        cnn.add(LeakyReLU(0.2))
        cnn.add(Dropout(0.3))

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

    def build_generator(self, latent_size):
        num_classes = 10
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

    def process(self):
        latent_size = 100

        self.g = self.build_generator(latent_size)

        # load the weights from the last epoch
        self.g.load_weights(self.last_epoch_path)

        np.random.seed(31337)

        noise = np.tile(np.random.uniform(-1, 1, (10, latent_size)), (10, 1))
        sampled_labels = np.array([
            [i] * 10 for i in range(10)
        ]).reshape(-1, 1)

        # get a batch to display
        generated_images = self.g.predict(
            [noise, sampled_labels], verbose=0)

        # arrange them into a grid
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(generated_images, 10)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        plt.imshow(img, cmap='gray')  # , interpolation='nearest')
        _ = plt.axis('off')

    def make_digit(self, digit=None):
        latent_size = 100
        noise = np.random.uniform(-1, 1, (1, latent_size))

        sampled_label = np.array([
            digit if digit is not None else np.random.randint(0, 10, 1)
        ]).reshape(-1, 1)

        generated_image = self.g.predict(
            [noise, sampled_label], verbose=0)

        return np.squeeze((generated_image * 127.5 + 127.5).astype(np.uint8))

    def _predict(self, num):
        self.process()
        # plt.imshow(make_digit(digit=8), cmap='gray_r', interpolation='nearest')
        # _ = plt.axis('off')
        return self.make_digit(digit=num)
