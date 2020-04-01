# -*- coding: utf-8 -*-

import yaml
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
from sklearn.cross_validation import train_test_split
import multiprocessing
import numpy as np
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml
from sklearn.preprocessing import LabelEncoder

np.random.seed(1337)  # For Reproducibility
import jieba
import pandas as pd

sys.setrecursionlimit(1000000)
# set parameters:
vocab_dim = 100
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 10
window_size = 7
batch_size = 32
n_epoch = 15
input_length = 100
cpu_count = multiprocessing.cpu_count()


# 加载训练文件

def loadfile():
    fopen = open('data/question_query.txt', 'r')
    questtion = []
    for line in fopen:
        question.append(line)

    fopen = open('data/music_query.txt', 'r')
    music = []
    for line in fopen:
        music.append(line)

    fopen = open('data/station_query.txt', 'r')
    station = []
    for line in fopen:
        station.append(line)

    combined = np.concatenate((station, music, qabot))
    question_array = np.array([-1] * len(question), dtype=int)
    station_array = np.array([0] * len(station), dtype=int)
    music_array = np.array([1] * len(music), dtype=int)
    # y = np.concatenate((np.ones(len(station), dtype=int), np.zeros(len(music), dtype=int)),qabot_array[0])
    y = np.hstack((qabot_array, station_array, music_array))
    print
    "y is:"
    print
    y.size
    print
    "combines is:"
    print
    combined.size
    return combined, y


# 对句子分词，并去掉换行符
def tokenizer(document):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    # text = [jieba.lcut(document.replace('\n', '')) for str(document) in text_list]
    result_list = []
    for text in document:
        result_list.append(' '.join(jieba.cut(text)).encode('utf-8').strip())
    return result_list


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
        4- 返回所有词语的向量的拼接结果
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data = []
            for sentence in combined:
                new_txt = []
                sentences = sentence.split(' ')
                for word in sentences:
            try:
                word = unicode(word, errors='ignore')
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)
            data.append(new_txt)

        return data
    combined = parse_dataset(combined)
    combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
    return w2indx, w2vec, combined

else:
print
'No data provided...'


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
    # 加载word2vec 模型
    model = Word2Vec.load('lstm_data/model/Word2vec_model.pkl')
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined


def get_data(index_dict, word_vectors, combined, y):
    # 获取句子的向量
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    # encode class values as integers
    encoder = LabelEncoder()
    encoded_y_train = encoder.fit_transform(y_train)
    encoded_y_test = encoder.fit_transform(y_test)
    # convert integers to dummy variables (one hot encoding)
    y_train = np_utils.to_categorical(encoded_y_train)
    y_test = np_utils.to_categorical(encoded_y_test)
    print
    x_train.shape, y_train.shape
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


##定义网络结构
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    nb_classes = 3
    print
    'Defining a Simple Keras Model...'
    ## 定义基本的网络结构
    model = Sequential()  # or Graph or whatever
    ## 对于LSTM 变长的文本使用Embedding 将其变成指定长度的向量
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    ## 使用单层LSTM 输出的向量维度是50，输入的向量维度是vocab_dim,激活函数relu
    model.add(LSTM(output_dim=50, activation='relu', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    ## 在这里外接softmax，进行最后的3分类
    model.add(Dense(output_dim=nb_classes, input_dim=50, activation='softmax'))
    print
    'Compiling the Model...'
    ## 激活函数使用的是adam
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print
    "Train..."
    print
    y_train
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch, verbose=1, validation_data=(x_test, y_test))
    print
    "Evaluate..."
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size)
    yaml_string = model.to_yaml()
    with open('lstm_data/lstm_koubei.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('lstm_data/lstm_koubei.h5')
    print
    'Test score:', score


# 训练模型，并保存
def train():
    print
    'Loading Data...'
    combined, y = loadfile()
    print
    len(combined), len(y)
    print
    'Tokenising...'
    combined = tokenizer(combined)
    print
    'Training a Word2vec model...'
    index_dict, word_vectors, combined = word2vec_train(combined)
    print
    'Setting up Arrays for Keras Embedding Layer...'
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
    print
    x_train.shape, y_train.shape
    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


# 训练模型，并保存
def self_train():
    print
    'Loading Data...'
    combined, y = loadfile()
    print
    len(combined), len(y)
    print
    'Tokenising...'
    combined = tokenizer(combined)
    print
    'Training a Word2vec model...'
    index_dict, word_vectors, combined = word2vec_train(combined)
    print
    'Setting up Arrays for Keras Embedding Layer...'
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
    print
    x_train.shape, y_train.shape
    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


def input_transform(string):
    words = ' '.join(jieba.cut(string)).encode('utf-8').strip()
    tmp_list = []
    tmp_list.append(words)
    # words=np.array(tmp_list).reshape(1,-1)
    model = Word2Vec.load('lstm_data/model/Word2vec_model.pkl')
    _, _, combined = create_dictionaries(model, tmp_list)
    return combined248


if __name__ == '__main__':
    self_train()