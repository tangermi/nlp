# -*- coding: utf-8 -*-
import sys
import os
from pyltp import Segmentor



model_path = "/home/xiaoxinwei/data/ltp_data_v3.4.0/cws.model" #Ltp3.3 分词库
segmentor = Segmentor()
segmentor.load(model_path)

words = segmentor.segment("在包含问题的所有解的解空间树中，按照深度优先搜索的策略，从根结点出发深度探索解空间树。")
print(" | ".join(words)) # 分割后的分词结果
