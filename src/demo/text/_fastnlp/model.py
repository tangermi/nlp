# -*- coding:utf-8 -*-
from fastNLP import Trainer
# from copy import deepcopy
from fastNLP.core.losses import CrossEntropyLoss
from fastNLP.core.optimizer import Adam
from fastNLP.core.utils import _save_model
from DPCNN import *
from fastNLP import Tester
from fastNLP.core.metrics import AccuracyMetric
from DPCNN import *
import torch
import os


# 读取模型和测试
class Model:
    @staticmethod
    def train_and_save_model(data_train, data_test, vocab, max_sentence_length, save_dir):
        # 确认torch版本是否能与fastnlp兼容
        print(torch.__version__)

        # 设置超参
        word_embedding_dimension = 300
        num_classes = 5

        # 读取神经网络
        model = DPCNN(max_features=len(vocab), word_embedding_dimension=word_embedding_dimension,
                      max_sentence_length=max_sentence_length, num_classes=num_classes)

        # 定义 loss 和 metric
        loss = CrossEntropyLoss(pred="output", target="label_seq")
        metric = AccuracyMetric(pred="predict", target="label_seq")

        # train model with train_data,and val model witst_data
        # embedding=300 gaussian init，weight_decay=0.0001, lr=0.001，epoch=5
        trainer = Trainer(model=model, train_data=data_train, dev_data=data_test, loss=loss, metrics=metric,
                          save_path='CD',
                          batch_size=64, n_epochs=5, optimizer=Adam(lr=0.001, weight_decay=0.0001))
        trainer.train()
        # 存储模型
        _save_model(model, model_name='new_model.pkl', save_dir=save_dir)

    @staticmethod
    def ensure_model(model_name, save_dir):
        model_path = os.path.join(save_dir, model_name)
        if model_path.is_file():
            model = torch.load(model_path)
            return model
        else:  # 如果模型不存在
            print('模型尚未训练')

    @staticmethod
    def load_model(save_dir, model_name='new_model.pkl'):
        model_path = os.path.join(save_dir, model_name)
        return torch.load(model_path)

    @staticmethod
    def test_model(data_test, model):
        # 使用tester来进行测试
        tester = Tester(data=data_test, model=model, metrics=AccuracyMetric(pred="predict", target="label_seq"),
                        batch_size=4)
        acc = tester.test()
        print(acc)
