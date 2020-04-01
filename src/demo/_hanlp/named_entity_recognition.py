# -*- coding:utf-8 -*-
from pyhanlp import *


class Entity_Recognition:
    # 中国人名识别
    def Segment_with_chinese_name(self, sentence):
        segment = HanLP.newSegment().enableNameRecognize(True)
        return segment.seg(sentence)

    # 音译人名识别
    def Segment_with_translated_name(self, sentence):
        segment = HanLP.newSegment().enableTranslatedNameRecognize(True)
        return segment.seg(sentence)

    # 日本人名识别
    def Segment_with_jp_name(self, sentence):
        segment = HanLP.newSegment().enableJapaneseNameRecognize(True)
        return segment.seg(sentence)

    # 地名识别
    def Segment_with_place_name(self, sentence):
        segment = HanLP.newSegment().enablePlaceRecognize(True)
        return segment.seg(sentence)

    # 机构名识别
    def Segment_with_org_name(self, sentence):
        segment = HanLP.newSegment().enableOrganizationRecognize(True)
        return segment.seg(sentence)

# test
if __name__ == '__main__':
    entity_recog = Entity_Recognition()
    # 中国人名识别
    sentence_for_chinese_name = "签约仪式前，秦光荣、李纪恒、仇和等一同会见了参加签约的企业家。"
    print('中国人名识别' + '-' * 20)
    print(entity_recog.Segment_with_chinese_name(sentence_for_chinese_name))
    # 音译人名识别
    sentence_for_translated_name = "一桶冰水当头倒下，微软的比尔盖茨、Facebook的扎克伯格跟桑德博格、亚马逊的贝索斯、苹果的库克全都不惜湿身入镜，这些硅谷的科技人，飞蛾扑火似地牺牲演出，其实全为了慈善。"
    print('音译人名识别' + '-' * 20)
    print(entity_recog.Segment_with_translated_name(sentence_for_translated_name))
    # 日本人名识别
    sentence_for_jp_name = "北川景子参演了林诣彬导演的《速度与激情3》"
    print('日本人名识别' + '-' * 20)
    print(entity_recog.Segment_with_jp_name(sentence_for_jp_name))
    # 地名识别
    sentence_for_place_name = "蓝翔给宁夏固原市彭阳县红河镇黑牛沟村捐赠了挖掘机"
    print('地名识别' + '-' * 20)
    print(entity_recog.Segment_with_place_name(sentence_for_place_name))
    # 机构名识别
    sentence_for_org_name = "我在上海林原科技有限公司兼职工作，偶尔去地中海影城看电影。"
    print('机构名识别' + '-' * 20)
    print(entity_recog.Segment_with_org_name(sentence_for_org_name))
