# -*- coding:utf-8 -*-
import jieba


# 使用jieba进行分词
def segment(train_path):
    f = open(train_path, 'r', encoding='utf-8')
    all_str = f.read().replace('\n', '').replace(' ', '')  # 去除空格
    f.close()
    cut_list = jieba.cut(all_str)
    seg_list = []  # 分词后的文本数据
    for c in cut_list:
        seg_list.append(c)
    return seg_list
