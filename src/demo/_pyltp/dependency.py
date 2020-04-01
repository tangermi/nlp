# -*- coding:utf-8 -*-
import os

from pyltp_segment import PyltpSegment
from pyltp import Parser


class DependencyParser(PyltpSegment):
    def parse(self, text):
        par_model_path = os.path.join(self.ltp_data_dir, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

        parser = Parser()  # 初始化实例
        parser.load(par_model_path)  # 加载模型

        words, postags = self.cut(text, part_of_speech=True)

        arcs = parser.parse(words, postags)  # 句法分析
        # parse结果的输出
        # print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))

        parser.release()  # 释放模型
        return arcs

if __name__ == '__main__':
    test_string = '距离除夕越来越近，这几天，铁路部门迎来了返乡客流的最高峰。然而，仍有些人，' \
                  '为买不到返乡的车票发愁。某些热门线路，为什么12306总是显示没票可卖?' \
                  '第三方抢票软件到底能不能顺利抢到票?铁路售票平台有没有更多的渠道向社会开放?'

    parser = DependencyParser()
    words, postags = parser.cut(test_string, part_of_speech=True)
    arcs = parser.parse(test_string)
    for i, word in enumerate(words):
        print(word, arcs[i].head, arcs[i].relation)
