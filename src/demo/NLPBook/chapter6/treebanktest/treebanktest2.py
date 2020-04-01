# -*- coding: utf-8 -*-
import sys, os
import re
from treelib import *
"""
得到相应的子树
"""

mytree = Tree.fromstring(u"(IP (IP (NP-SBJ (NN 建筑)  (NN 公司))  (VP (VV 进)    (NP-OBJ (NN 区)))) (PU ，)  (IP (NP-SBJ (ADJP (JJ 有关))  (NP (NN 部门)))  (VP (ADVP (AD 先))     (VP (VV 送上)  (NP-OBJ (DP (DT 这些))       (NP (NN 法规性)  (NN 文件))))))   (PU ，)   (IP (NP-SBJ (-NONE- *pro*))   (VP (ADVP (AD 然后))         (VP (VE 有)  (IP-OBJ (NP-SBJ (ADJP (JJ 专门))  (NP (NN 队伍)))   (VP (VV 进行)     (NP-OBJ (NN 监督)     (NN 检查)))))))  (PU 。))  ")
# 找到"公司"所在的IP子树
branch = getbranch(mytree,"公司","IP")
print(str(branch))