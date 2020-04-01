# -*- coding:utf-8 -*-
from pyhanlp import *


class Pinyin:
    def __init__(self, sentence):
        self.pinyin_list = HanLP.convertToPinyinList(sentence)

    # 获取拼音数字音调
    def get_tone_num(self):
        pinyin = ''
        for each in self.pinyin_list:
            pinyin += str(each) + ' '
        return pinyin

    # 获取符号音调
    def get_tone_mark(self):
        tone = ''
        for pinyin in self.pinyin_list:
            tone += str(pinyin.getPinyinWithToneMark()) + ' '
        return tone

    # 无音调
    def get_pinyin(self):
        pinyin = ''
        for each in self.pinyin_list:
            pinyin += str(each.getPinyinWithoutTone()) + ' '
        return pinyin

    # 声调
    def get_Tone(self):
        tone = ''
        for pinyin in self.pinyin_list:
            tone += str(pinyin.getTone()) + ' '
        return tone

    # 声母
    def get_shengmu(self):
        shengmu = ''
        for pinyin in self.pinyin_list:
            shengmu += str(pinyin.getShengmu()) + ' '
        return shengmu

    # 韵母
    def get_yunmu(self):
        yunmu = ''
        for pinyin in self.pinyin_list:
            yunmu += str(pinyin.getYunmu()) + ' '
        return yunmu

    # 输入法头
    def get_head(self):
        head = ''
        for pinyin in self.pinyin_list:
            head += str(pinyin.getHead()) + ' '
        return head


if __name__ == '__main__':
    pinyin = Pinyin('拼音转换')
    # 获取拼音数字音调
    print(pinyin.get_tone_num())

    # 获取符号音调
    print(pinyin.get_tone_mark())

    # 无音调
    print(pinyin.get_pinyin())

    # 声调
    print(pinyin.get_Tone())

    # 声母
    print(pinyin.get_shengmu())

    # 韵母
    print(pinyin.get_yunmu())

    # 输入法头
    print(pinyin.get_head())

