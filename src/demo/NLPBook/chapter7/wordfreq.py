# -*- coding: utf-8 -*-
"""
基于PFR的词频统计的实现
"""
import sys  
import os
import traceback  
import time
from framework import *
import nltk
import re
import io



def fullfreq(fcontent,sumdict):
	sent = " ".join(fcontent.splitlines()).strip().decode("utf8")
	sTuple=[ nltk.tag.str2tuple(t) for t in sent.split(" ") ]
	fredist=nltk.FreqDist(sTuple) #获取统计结果
	print(len(fredist))
	for localkey in fredist:
		if localkey in sumdict : #检查当前词频是否在字典中存在
			sumdict[localkey]=sumdict[localkey]+fredist[localkey] #如果存在，将词频累加，并更新字典值
		elif str(localkey[1]).find("None")==-1 :
			sumdict[localkey]=fredist[localkey] #将当前词频添加到字典中	

hanzi= re.compile(r"[\u4e00-\u9fa5]+") #切分汉字
def hanzfreq(fcontent,sumdict):
	sent = " ".join(fcontent.splitlines()).strip()
	sTuple=[ nltk.tag.str2tuple(t) for t in sent.split(" ") if hanzi.match(t) ]
	fredist=nltk.FreqDist(sTuple) #获取统计结果
	print(len(fredist))
	for localkey in fredist:
		if localkey in sumdict : #检查当前词频是否在字典中存在
			sumdict[localkey]=sumdict[localkey]+fredist[localkey] #如果存在，将词频累加，并更新字典值
		elif str(localkey[1]).find("None")==-1 : 
			sumdict[localkey]=fredist[localkey] #将当前词频添加到字典中		

sumdict={}	# 统计结果
rootdir = "/home/xiaoxinwei/data/199801/"
segpath = "199801.txt"
with open(rootdir + segpath,'r', encoding='gbk', errors ='ignore') as f:
	segcorpus = f.readlines()
# fullfreq(segcorpus,sumdict) # 包含非汉字的词频统计
hanzfreq(str(segcorpus),sumdict) # 仅有汉字的词频统计
sumlist= sorted(sumdict.items(), key=lambda x:x[1],reverse=True)
file_str = io.StringIO()
for key in sumlist:
	file_str.write(str(key[0][0]));	file_str.write("\t")
	file_str.write(str(key[0][1]));	file_str.write("\t")	
	file_str.write(str(key[1]));	file_str.write("\n")
savefile(rootdir+"freqdict.txt",file_str.getvalue()) #freqdict.txt
print("ok")