# -*- coding:utf-8 -*-
from iso639_2 import iso_639_choices_dict, eng2chinese_dict
from langid.langid import LanguageIdentifier, model

import langid


def ISOcode2lang(code, iso_639_choices_dict = iso_639_choices_dict):
    lang = iso_639_choices_dict[code[0]]
    if lang in eng2chinese_dict:
        lang = eng2chinese_dict[lang]
    return lang

if __name__ == '__main__':
    text = '今天de天气shi多云转阴'
    text2 = '我吃个apple会变得healthy,你吃个banana会变得retard'

    # 输出iso639-1（国际标准化组织语言编码标准第一部分） 以及unnormalized probability estimate，未归一化的概率估计
    print(langid.classify(text))
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    # 输出iso639-1（国际标准化组织语言编码标准第一部分） 以及归一化的概率估计
    print(identifier.classify(text))

    # 输出检测语言的汉语名，比如“简体中文”，“德语”
    print(ISOcode2lang(langid.classify(text)))
    print(ISOcode2lang(langid.classify(text2)))
    # 可以为输出语言设置限制，比如把限制设为德语，法语，和意大利语
    langid.set_languages(['de','fr','it'])
    print(langid.classify(text))  # 这里将很明显的中文文档识别为法语，由于这个限制



