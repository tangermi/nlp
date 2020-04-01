# -*- coding:utf-8 -*-
img_path = "/apps/data/ai_nlp_testing/1.png"

###
from matplotlib import pyplot as plt
# from PIL import Image
# img = Image.open(img_path)
# # plt.figure("dog")
# plt.imshow(img)
# plt.show()



# 尝试在pycharm上显示图片
from PIL import Image
img = Image.open('price2item.png')
plt.imshow(img)
plt.show()

