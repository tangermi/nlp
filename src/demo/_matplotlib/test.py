# -*- coding:utf-8 -*-
img_path = "/apps/data/ai_nlp_testing/1.png"

###
from matplotlib import pyplot as plt
from PIL import Image
img = Image.open(img_path)
# plt.figure("dog")
plt.imshow(img)
plt.show()


