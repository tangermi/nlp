# -*- coding:utf-8 -*-
import matplotlib
matplotlib.use('Agg')

from IPython.display import Image, display
from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget

cf = CanvasFrame()
# t = Tree.fromstring('(S (NP this tree) (VP (V is) (AdjP pretty)))')
# tc = TreeWidget(cf.canvas(), t)
# cf.add_widget(tc, 10, 10)   # (10,10) offsets
# cf.print_to_file('tree.ps')
# cf.destroy()

# import os
# os.system('convert output.ps output.png')
