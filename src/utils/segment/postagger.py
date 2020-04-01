# -*- coding:utf-8 -*-
from pyhanlp import *


# 词性标注
class Postagger:
    @staticmethod
    # 传入参数选择分词模式： -h为HanLP分词, -j为jieba分词， -p为pyltp分词
    def cut(sentence, mode):
        if mode == '-h':
            term_list = HanLP.segment(sentence)
            result = [[term.word, str(term.nature)] for term in term_list]
            return result
        if mode == '-j':
            import jieba.posseg
            seg_list = jieba.posseg.cut(sentence)
            result = [[word, flag] for word, flag in seg_list]
            return result
        if mode == '-p':
            from pyltp import Segmentor
            from pyltp import Postagger
            cws_model_path = '/apps/data/ai_nlp_testing/model/ltp_data_v3.4.0/cws.model'
            segmentor = Segmentor()
            segmentor.load(cws_model_path)
            words = segmentor.segment(sentence)
            segmentor.release()
            pos_model_path = '/apps/data/ai_nlp_testing/model/ltp_data_v3.4.0/pos.model'
            postagger = Postagger()
            postagger.load(pos_model_path)
            postags = postagger.postag(words)
            postagger.release()
            result = []
            for number, y in enumerate(words):
                result.append([words[number], postags[number]])
            return result


if __name__ == '__main__':
    print(Postagger.cut("我不是流言，不能猜测你", '-h'))
    print(Postagger.cut("我不是流言，不能猜测你", '-j'))
    print(Postagger.cut("我不是流言，不能猜测你", '-p'))
