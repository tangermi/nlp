# -*- coding:utf-8 -*-
from pyltp import Segmentor


class PyltpSegment:
    def __init__(self):
        self.segmentor = Segmentor()  # 初始化实例
        self.user_dict = '/apps/data/ai_nlp_testing/raw/customized_dict_for_ltp/chemistry.txt'
        self.ltp_data_dir = '/apps/data/ai_nlp_testing/model/ltp_data_v3.4.0'

    # 分句
    def cut_sentence(self, doc):
        from pyltp import SentenceSplitter
        return SentenceSplitter.split(doc)

    #分词
    def cut(self, text, part_of_speech=False):
        import os
        cws_model_path = os.path.join(self.ltp_data_dir, 'cws.model')

        self.segmentor.load(cws_model_path)
        words = list(self.segmentor.segment(text))
        self.segmentor.release()
        if not part_of_speech:
            return words
        else:
            from pyltp import Postagger
            pos_model_path = os.path.join(self.ltp_data_dir, 'pos.model')
            postagger = Postagger()  # 初始化实例
            postagger.load(pos_model_path)  # 加载模型

            words = list(words)  # 分词结果
            postags = list(postagger.postag(words))  # 词性标注结果
            postagger.release()  # 释放模型
            return words, postags
            # result = []
            # for number, y in enumerate(words):
            #     result.append([words[number], postags[number]])
            # return result


# test
if __name__ == '__main__':
    test_string = '距离除夕越来越近，这几天，铁路部门迎来了返乡客流的最高峰。然而，仍有些人，' \
                  '为买不到返乡的车票发愁。某些热门线路，为什么12306总是显示没票可卖?' \
                  '第三方抢票软件到底能不能顺利抢到票?铁路售票平台有没有更多的渠道向社会开放?'

    pyltpsegment = PyltpSegment()

    # 分句
    sents = pyltpsegment.cut_sentence(test_string)
    for sent in sents:
        print(sent)

    # 分词
    words = pyltpsegment.cut((test_string), part_of_speech=False)
    print(words)

    #词性标注
    words, postags = pyltpsegment.cut((test_string), part_of_speech=True)
    for i, word in enumerate(words):
        print(f'{word}/{postags[i]}', end=', ')

