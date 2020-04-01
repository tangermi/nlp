# -*- coding: utf-8 -*-
import sys  
import os
from stanford import StanfordPOSTagger



root = 'E:/nltk_data/stanford-corenlp/'
modelpath = root+"models/pos-tagger/chinese-distsim/chinese-distsim.tagger"
st = StanfordPOSTagger(root,modelpath)
propspath = "my.tagger.props"
st.genpropfile(propspath)
