# -*- coding: utf-8 -*-
"""
跑不通，因为没有数据，数据格式在P191
"""
import sys  
import os
import maxent

# # 设置 UTF-8输出环境
# reload(sys)
# sys.setdefaultencoding('utf-8')

model = maxent.MaxEnt()

model.load_data('data.txt')

model.train()

print(model.predict("Rainy Happy Dry"))
