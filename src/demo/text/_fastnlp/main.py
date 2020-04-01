# -*- coding:utf-8 -*-
from read_corpus import ReadCorpus
from preprocess import Preprocess
from model import Model
import time


def run():
    data_path = r'/apps/data/ai_nlp_testing/raw/training_data_for_classification/sogou_text_classification_corpus_mini'
    # 读取搜狗语料
    data_train, data_test = ReadCorpus.read_data(data_path)
    preprocess = Preprocess()
    # 对语料进行处理，以匹配fastNLP的输入结构
    data_train, data_test, vocab, max_sentence_length = preprocess.data_preprocess(data_train, data_test)

    # 模型存储路径
    save_dir = '/apps/data/ai_nlp_testing/model/sogou_corpus_fastNLP'
    print(len(data_train))
    print(len(data_test))
    # 训练模型 (如果已经训练好模型，可注释这一行，直接到下一步读取)
    Model.train_and_save_model(data_train, data_test, vocab, max_sentence_length, save_dir=save_dir)
    # 读取模型
    model = Model.load_model(save_dir=save_dir)
    # 测试模型
    Model.test_model(data_test, model)


# To do : 机器学习 需要更多内存
if __name__ == '__main__':
    start = time.clock()

    run()

    elapsed = (time.clock() - start)
    print(elapsed)
