# -*- coding:utf-8 -*-
from pyhanlp import *


class Sentiment_analyzer:
    def __init__(self):
        self.NaiveBayesClassifier = JClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
        self.IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')
        self.chn_senti_corp = '/apps/data/ai_nlp_testing/raw/training_data_for_sentiment_analysis/ChnSentiCorp_hotel_review'

    def predict(self, classifier, text):
        senti = self.sentimap(classifier.classify(text))
        return senti

    def train_or_load_classifier(self):
        model_path = self.chn_senti_corp + '.ser'
        if os.path.isfile(model_path):
            return self.NaiveBayesClassifier(self.IOUtil.readObjectFrom(model_path))
        classifier = self.NaiveBayesClassifier()
        classifier.train(self.chn_senti_corp)
        model = classifier.getModel()
        #  训练后的模型支持持久化，下次就不必训练了
        self.IOUtil.saveObjectTo(model, model_path)
        return self.NaiveBayesClassifier(model)

    def sentimap(self, senti):
        return {
            'negative': '负面',
            'positive': '正面',
        }.get(senti, 'error')


if __name__ == '__main__':
    testCase = [
        "前台客房服务态度非常好！早餐很丰富，房价很干净。再接再厉！",
        "结果大失所望，灯光昏暗，空间极其狭小，床垫质量恶劣，房间还伴着一股霉味。",
        "可利用文本分类实现情感分析，效果不是不行"
    ]
    sentiment_analyzer = Sentiment_analyzer()
    classifier = sentiment_analyzer.train_or_load_classifier()
    senti = sentiment_analyzer.predict(classifier, testCase[0])
    print(f"《{testCase[0]}》 情感极性是 【{senti}】")
    senti = sentiment_analyzer.predict(classifier, testCase[1])
    print(f"《{testCase[1]}》 情感极性是 【{senti}】")
    senti = sentiment_analyzer.predict(classifier, testCase[2])
    print(f"《{testCase[2]}》 情感极性是 【{senti}】")
