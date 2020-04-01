# -*- coding:utf-8 -*-
from text_classification import Text_classifier


def run():
    testCase = [
        'C罗获2018环球足球奖最佳球员 德尚荣膺最佳教练',
        '英国造航母耗时8年仍未服役 被中国速度远远甩在身后',
        '研究生考录模式亟待进一步专业化',
        '如果真想用食物解压,建议可以食用燕麦',
        '通用及其部分竞争对手目前正在考虑解决库存问题',
        '归化球员救不了中国足球',
    ]

    data_path = '/apps/data/ai_nlp_testing/raw/training_data_for_classification/sogou_text_classification_corpus_mini'
    text_classifier = Text_classifier(data_path)
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


# text
if __name__ == '__main__':
    run()
