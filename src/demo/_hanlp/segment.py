# -*- coding:utf-8 -*-
from pyhanlp import *
import time


class Segment:
    # HMM-Bigram（速度与精度最佳平衡，一百兆内存）
    # N-最短路径分词
    def n_shortest_segment(self, sentence):
        NShortSegment = JClass("com.hankcs.hanlp.seg.NShort.NShortSegment")
        n_short_segment = NShortSegment().enableCustomDictionary(False).enablePlaceRecognize(
            True).enableOrganizationRecognize(True)
        return n_short_segment.seg(sentence)



    # 最短路分词
    def shortest_segment(self, sentence):
        DijkstraSegment = JClass("com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment")
        shortest_segment = DijkstraSegment().enableCustomDictionary(False).enablePlaceRecognize(
            True).enableOrganizationRecognize(True)
        return shortest_segment.seg(sentence)

    # 由字构词（侧重精度，全世界最大语料库，可知别新词，适合NLP任务）
    # CRF分词
    def crf_analyze(self, sentence):
        CRFLexicalAnalyzer = JClass("com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer")
        analyzer = CRFLexicalAnalyzer()
        return analyzer.analyze(sentence)

    # 词典分词 （侧重速度，每秒数千万字符；省内存）
    # 极速词典分词
    def highSpeedSegment(self, sentence):
        SpeedTokenizer = JClass("com.hankcs.hanlp.tokenizer.SpeedTokenizer")
        return SpeedTokenizer.segment(sentence)


    # 索引分词
    def indexTokenize(self, document):
        IndexTokenizer = JClass('com.hankcs.hanlp.tokenizer.IndexTokenizer')
        return IndexTokenizer.segment(sentence)


# test
if __name__ == '__main__':
    testCases = [
        "商品和服务",
        "结婚的和尚未结婚的确实在干扰分词啊",
        "买水果然后来世博园最后去世博会",
        "中国的首都是北京",
        "欢迎新老师生前来就餐",
        "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
        "随着页游兴起到现在的页游繁盛，依赖于存档进行逻辑判断的设计减少了，但这块也不能完全忽略掉。"]
    document = "水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，" \
               "根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，" \
               "有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，" \
               "严格地进行水资源论证和取水许可的批准。"
    segment = Segment()
    # N-最短路径分词
    print('N-最短路径分词' + '-' * 20)
    for sentence in testCases:
        print(f'N-最短分词：{segment.n_shortest_segment(sentence)}')

    # 最短路径分词
    print('最短路径分词' + '-' * 20)
    for sentence in testCases:
        print(f'最短路分词：{segment.shortest_segment(sentence)}')

    # CRF分词
    print('CRF分词' + '-' * 20)
    for sentence in testCases:
        print(segment.crf_analyze(sentence))


    # 极速词典分词
    print('极速词典分词' + '-' * 20)
    if True:
        start = time.time()
        pressure = 100000
        for i in range(pressure):
            segment.highSpeedSegment(testCases[-1])
        cost_time = time.time() - start
        print("SpeedTokenizer分词速度：%.2f字每秒" % (len(sentence) * pressure / cost_time))

    # 索引分词
    print('索引分词' + '-' * 20)
    for sentence in testCases:
        print(segment.indexTokenize(testCases))


