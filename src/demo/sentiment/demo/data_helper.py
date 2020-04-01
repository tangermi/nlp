import  os
import re
import numpy as np
import jieba # 结巴分词
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


def loadFile_tokenize(cn_model):
    pos_txts = os.listdir('data/pos')
    neg_txts = os.listdir('data/neg')
    train_texts_orig = []
    for i in range(len(pos_txts)):
        with open('data/pos/'+pos_txts[i],'r',encoding='gbk',errors="ignore") as f:
            text = f.read().strip()
            train_texts_orig.append(text)
    for  i in range(len(neg_txts)):
        with open("data/neg/"+neg_txts[i],'r',encoding='gbk',errors="ignore") as  f:
            text = f.read().strip()
            train_texts_orig.append(text)
    train_tokens = []
    for text in train_texts_orig:
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
        cut = jieba.cut(text)
        cut_list = [ i for i in cut ]
        for i,word in enumerate(cut_list):
            try:
                cut_list[i] = cn_model.vocab[word].index
            except KeyError:
                print('Keyerror')
                cut_list[i] = 0
        train_tokens.append(cut_list)
    return train_tokens
    
def tokenize(train_texts_orig):
    train_tokens = []
    for text in train_texts_orig:
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
        cut = jieba.cut(text)
        cut_list = [ i for i in cut ]
        for i,word in enumerate(cut_list):
            try:
                #将词转换为索引
                cut_list[i] = cn_model.vocab[word].index
            except:
                cut_list[i] = 0
        train_tokens.append(cut_list)
    return train_tokens

def getMaxTokens(train_tokens):
    num_tokens = [ len(tokens) for tokens in train_tokens ]
    num_tokens = np.array(num_tokens)
    max_tokens = np.mean(num_tokens) + 4 * np.std(num_tokens)
    max_tokens = int(max_tokens)
    print("max_tokens={}".format(max_tokens))
    return max_tokens

def buildEmbeddingLayer(cn_model):
    num_words = 50000
    embedding_dim = 300
    # 初始化embedding_matrix，之后在keras上进行应用
    embedding_matrix = np.zeros((num_words, embedding_dim))
    # embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
    # 维度为 50000 * 300
    for i in range(num_words):
        embedding_matrix[i,:] = cn_model[cn_model.index2word[i]]
    embedding_matrix = embedding_matrix.astype('float32')
    return embedding_matrix

def getData(train_tokens,max_tokens):
    train_pad = pad_sequences(train_tokens, maxlen=max_tokens, padding='pre', truncating='pre')
    train_target = np.concatenate((np.ones(2000),np.zeros(2000)))
    X_train, X_test, y_train, y_test = train_test_split(train_pad,train_target,test_size=0.1,random_state=213)
    return X_train, X_test, y_train, y_test