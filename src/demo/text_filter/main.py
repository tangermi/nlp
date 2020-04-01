# -*- coding:utf-8 -*-
from NaiveFilter import  NaiveFilter
from DFAFilter import  DFAFilter
from BSFilter import  BSFilter


if __name__ == '__main__':
    # gfw = NaiveFilter()
    # gfw = BSFilter()
    gfw = DFAFilter()
    gfw.parse(r"/apps/data/ai_nlp_testing/raw/sensored_words/keywords")
    import time

    t = time.time()
    print(gfw.filter("法轮功 shabi", "*"))
    print(gfw.filter("针孔摄像机 caonima", "*"))
    print(gfw.filter("售假人民币 操你妈", "*"))
    print(gfw.filter("传世私服 操场我操操操", "*"))
    print(gfw.filter("您吃了没", "*"))
    print(f'运行时间: {time.time() - t}毫秒')
