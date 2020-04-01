# -*- coding: utf-8 -*-
import pickle
import h5py
import numpy as np


class DataReader:
    @staticmethod
    def read_file(path):
        with open(path, 'r') as f:
            content = f.read()
        return content

    @staticmethod
    def read_file_raw(path):
        with open(path, 'r') as f:
            content = f.read().decode('utf-8')
        return content

    @staticmethod
    def write_file(save_path, content):
        with open(save_path, 'w') as f1:
            f1.write(content.encode('utf-8'))

    @staticmethod
    def load_stopwords(stop_words_file):
        with open(stop_words_file, 'rb') as f:
            stopwords = f.read().decode('GBK')
        stopwords_list = stopwords.split('\r\n')
        stopwords_list = [i for i in stopwords_list]
        return stopwords_list

    @staticmethod
    def read_bunch(bunch_path):
        with open(bunch_path, 'rb') as f:
            bunch = pickle.load(f)
        return bunch

    @staticmethod
    def write_bunch(path, bunchobj):
        with open(path, 'wb') as f:
            pickle.dump(bunchobj, f)

    @staticmethod
    def h5_train_set(train_path):
        with h5py.File(train_path, 'r') as f:
            train_set_x = np.array(f["train_set_x"][:])
            train_set_y = np.array(f["train_set_y"][:])

        train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))

        train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], -1).T

        # 对数据集进行居中和标准化
        train_set_x = train_set_x_flatten / 255.

        return train_set_x, train_set_y

    @staticmethod
    def h5_test_set(test_path):
        with h5py.File(test_path, 'r') as f:
            test_set_x = np.array(f["test_set_x"][:])
            test_set_y = np.array(f["test_set_y"][:])
            classes = np.array(f["list_classes"][:])

        test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

        test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T

        # 对数据集进行居中和标准化
        test_set_x = test_set_x_flatten / 255.

        return test_set_x, test_set_y, classes
