# -*- coding:utf-8 -*-
from nltk.tree import Tree
from IPython.display import display
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
from matplotlib import pyplot as plt
from PIL import Image
import os
from nltk.draw.tree import TreeView


# cf = CanvasFrame()
tree = Tree.fromstring('(S (NP this tree) (VP (V is) (AdjP pretty)))')
TreeView(tree)._cframe.print_to_file('tree.ps')
# tc = TreeWidget(cf.canvas(),t)
# cf.add_widget(tc, 10, 10)
# cf.print_to_file('tree.ps')
# os.system('convert tree.cs tree.png')
# img = Image.open(tree.ps)
# plt.imshow(img)
# plt.show()
# display(tree)
