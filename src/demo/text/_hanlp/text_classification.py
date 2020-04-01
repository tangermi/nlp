# -*- coding:utf-8 -*-
import os
from pyhanlp import SafeJClass


class Text_classifier:
    def __init__(self, data_path):
        self.NaiveBayesClassifier = SafeJClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
        self.IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')
        # 使用搜狗文本分类语料库迷你版，包含５个类别，每个类别１０００条数据。
        self.data_path = data_path

    def train_or_load_classifier(self):
        model_path = self.data_path + '.ser'
        if os.path.isfile(model_path):
            return self.NaiveBayesClassifier(self.IOUtil.readObjectFrom(model_path))
        classifier = self.NaiveBayesClassifier()
        classifier.train(self.data_path)
        model = classifier.getModel()
        self.IOUtil.saveObjectTo(model, model_path)
        return self.NaiveBayesClassifier(model)

    def predict(self, classifier, text):
        category = self.classmap(classifier.classify(text))
        return category
        # 如需获取离散型随机变量的分布，请使用predict接口
        # print(f'《{text}》\t属于分类\t【{classifier.predict(text)}】')

    def classmap(self, num):
        # 后期可用单独文件存储map，以适配更多分类以及更大数据.
        return {
            0: '体育',
            1: '健康',
            2: '军事',
            3: '教育',
            4: '汽车',
        }.get(int(num), 'error')
