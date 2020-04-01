# -*- coding: utf-8 -*-
"""
this is a tool file .It has load_file,save_file,logistic,softmax function
"""
import pickle
import numpy as np


def read_file(path):
    with open(path, 'rb') as f:
        file = f.read().decode('utf-8')
    return file


def writer_file(path, obj):
    with open(path, 'wb') as f:
        f.write(obj.encode('utf-8'))


def read_file_encode(path, encode):
    with open(path, 'rb') as f:
        file = f.read().decode(encode)
    return file


def writer_file_encode(path, obj, encode):
    with open(path, 'wb') as f:
        f.write(obj.encode(encode))


def read_stopwords(stop_words_file):
    with open(stop_words_file, 'r') as f:
        stopwords = f.read().decode('utf-8')
    stopwords_list = stopwords.split('\n')
    stopwords_list = [i for i in stopwords_list]
    return stopwords_list


def read_bunch(bunch_path):
    with open(bunch_path, 'rb') as f:
        bunch = pickle.load(f)
    return bunch


def write_bunch(path, bunchobj):
    with open(path, 'wb') as f:
        pickle.dump(bunchobj, f)


def read_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def write_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_model(path):
    with open(path, 'rb') as f:
        model_obj = pickle.load(f)
    return model_obj


def jieba_init(setting):
    if 'JIEBA' not in setting:
        return

    if not setting.get('isJieba'):
        return

    import fileinput
    import jieba
    import jieba.analyse
    dic_jieba = setting['JIEBA']
    if 'user_word' in dic_jieba:
        jieba.load_userdict(dic_jieba['user_word'])
        # jieba.add_word('路明非')

    if 'stop_word' in dic_jieba:
        jieba.analyse.set_stop_words(dic_jieba['stop_word'])

        with open(dic_jieba['stop_word']) as f:
            stopwords = filter(lambda x: x, map(lambda x: x.strip().decode('utf-8'), f.readlines()))
        stopwords.extend([' ', '\t', '\n'])
        dic_jieba['stop_words'] = frozenset(stopwords)

    if 'tag' in dic_jieba:
        tag_file = dic_jieba['tag']
        dic_jieba['tag'] = {}
        for line in fileinput.input(tag_file):
            line = line.strip("\n").strip("\r")
            if not line:
                continue

            word = line.split('\t')
            word[1] = word[1].decode('utf8')
            dic_jieba['tag'][word[1]] = word[0]


class Logistic():

    def sigmoid(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def init_w_b(self, dim):
        w = np.zeros((dim, 1))
        b = 0
        return w, b

    def propagate(self, w, b, X, Y):

        """
        X——（num，样本数）
        Y——（类别，样本数）
        cost —— logistic的似然计算出的损失值
        """

        m = X.shape[1]
        A = self.sigmoid(np.add(np.dot(w.T, X), b))
        cost = -(np.dot(Y, np.log(A).T) + np.dot(1 - Y, np.log(1 - A).T)) / m  # compute cost
        dw = np.dot(X, (A - Y).T) / m
        db = np.sum(A - Y) / m
        cost = np.squeeze(cost)

        grads = {"dw": dw,
                "db": db}

        return grads, cost

    def optimize(self, w, b, X, Y, num_iterations, learning_rate):
        costs = []

        for i in range(int(num_iterations)):
            grads, cost = self.propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]
            w = w - learning_rate * dw
            b = b - learning_rate * db
            if i % 100 == 0:
                costs.append(cost)
                self.logger.info("Cost after iteration %i: %f" % (i, cost))

        params = {"w": w,
                  "b": b}

        grads = {"dw": dw,
                 "db": db}

        return params, grads, costs


class Access_model():

    def dump_model(self, model_path, obj):
        with open(model_path, 'w') as f:
            pickle.dump(obj, f, protocol=2)

    def load_model(self, model_path):
        with open(model_path, 'r') as f:
            clf = pickle.load(f)
        return clf


# softmax的实现
class Softmax():

    """softmax
            parms: X: np.array(sample_nums, vector_dim)
                   y: np.array(label_nums, sample_nums)
                   w: np.array(label_nums, vector_dim)
                   loss: sum(-mul(y_true, log(A)))
                   grad: (y - y_hat)

    """

    def softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps)

    def stable_softmax(self, X):
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)

    def init_w(self, X):
        m = X.shape[1]
        w = np.random.uniform(0, 1, (10, m))
        return w

    def propagate(self, w, X, y):
        m = X.shape[1]
        A = self.stable_softmax(np.dot(w, X.T))

        # loss = -1/m * np.sum(np.log(A)*y)

        # logging.info(y.shape)
        # logging.info(A.shape)

        grad = -1/m * np.dot(y - A, X)
        return grad

    def optimize(self, w, X, y, num_iterations, learning_rate):
        costs = []

        for i in range(int(num_iterations)):
            grad = self.propagate(w, X, y)

            w = w - (learning_rate * grad)

            if i % 100 == 0:
                pass


def str2onehot(str, vocab):
    onehot = np.zeros((len(str), len(vocab)))
    for i, character in enumerate(str):
        index = vocab.find(character)
        onehot[i, index] = 1
    return onehot


def onehot2str(onehot, vocab):
    max_index = np.argmax(np.array(onehot), axis=1)
    str = []
    for i in range(max_index.shape[0]):
        character = ''.join([vocab[x] for x in max_index[i]])
        str.append(character)
    return str
