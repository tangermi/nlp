# -*- coding:utf-8 -*-
import os

from pyhanlp import SafeJClass


class Text_classifier:
    def __init__(self):
        self.NaiveBayesClassifier = SafeJClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
        self.IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')
        # 使用搜狗文本分类语料库迷你版，包含５个类别，每个类别１０００条数据。
        self.sogou_corpus_path = '/apps/data/ai_nlp_testing/raw/training_data_for_classification/sogou_text_classification_corpus_mini'

    def train_or_load_classifier(self):
        model_path = self.sogou_corpus_path + '.ser'
        if os.path.isfile(model_path):
            return self.NaiveBayesClassifier(self.IOUtil.readObjectFrom(model_path))
        classifier = self.NaiveBayesClassifier()
        classifier.train(self.sogou_corpus_path)
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
        }.get(int(num),'error')

# text
if __name__ == '__main__':
    testCase = [
        'C罗获2018环球足球奖最佳球员 德尚荣膺最佳教练',
        '英国造航母耗时8年仍未服役 被中国速度远远甩在身后',
        '研究生考录模式亟待进一步专业化',
        '如果真想用食物解压,建议可以食用燕麦',
        '通用及其部分竞争对手目前正在考虑解决库存问题',
        '归化球员救不了中国足球',
    ]
    text_classifier = Text_classifier()
    classifier = text_classifier.train_or_load_classifier()
    category = text_classifier.predict(classifier, testCase[0])
    print(f'《{testCase[0]}》\t属于分类\t【{category}】')
    category = text_classifier.predict(classifier, testCase[1])
    print(f'《{testCase[1]}》\t属于分类\t【{category}】')
    category = text_classifier.predict(classifier, testCase[2])
    print(f'《{testCase[2]}》\t属于分类\t【{category}】')
    category = text_classifier.predict(classifier, testCase[3])
    print(f'《{testCase[3]}》\t属于分类\t【{category}】')
    category = text_classifier.predict(classifier, testCase[4])
    print(f'《{testCase[4]}》\t属于分类\t【{category}】')
    category = text_classifier.predict(classifier, testCase[5])
    print(f'《{testCase[5]}》\t属于分类\t【{category}】')
