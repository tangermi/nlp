# -*- coding:utf-8 -*-


class Segmentor:
    @staticmethod
    # 传入参数选择分词模式： -h为HanLP分词, -j为jieba分词， -p为pyltp分词
    def cut(sentence, mode):
        if mode == '-h':
            from pyhanlp import HanLP
            term_list = HanLP.segment(sentence)
            return [term.word for term in term_list]
        if mode == '-j':
            import jieba
            seg_list = jieba.cut(sentence, cut_all=False)
            return list(seg_list)
        if mode == '-p':
            from pyltp import Segmentor
            cws_model_path = '/apps/data/ai_nlp_testing/model/ltp_data_v3.4.0/cws.model'
            segmentor = Segmentor()
            segmentor.load(cws_model_path)
            words = segmentor.segment(sentence)
            return list(words)


if __name__ == '__main__':
    print(Segmentor.cut("我不是流言，不能猜测你", '-h'))
    print(Segmentor.cut("我不是流言，不能猜测你", '-j'))
    print(Segmentor.cut("我不是流言，不能猜测你", '-p'))
