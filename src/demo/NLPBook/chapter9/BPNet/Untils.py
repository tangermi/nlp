# -*- coding: UTF-8 -*-
# Filename : BackPropgation.py
'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: jack zheng
'''
from numpy import *
import operator
import matplotlibp.pyplot as plt

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

# ���ݱ�׼��(��һ��):ͳ�ƾ�ֵ�ͱ�׼���һ��
def normalize(dataMat):
    # �����ֵ
    height = mean(dataMat[:,1])
    weight = mean(dataMat[:,2])	 
    # ���������
    stdh = std(dataMat[:,1])
    stdw = std(dataMat[:,2])
    # ��׼��
    dataMat[:,1] = (dataMat[:,1]-height)/stdh
    dataMat[:,2] = (dataMat[:,2]-weight)/stdw	 
    return dataMat	 

# ��ʾ����ͼ��
def displayplot():
    plt.show()	
    
# ���ƶ�ά���ݼ�����ɢ��ͼ:�޷���
# ������ List �� Matrix
def drawScatter(dataMat,flag=True):
    if type(dataMat) is list :
    	px = (mat(dataMat)[:,1]).tolist()
    	py = (mat(dataMat)[:,2]).tolist()	
    if type(dataMat) is matrix :
    	px = (dataMat[:,1]).tolist()
    	py = (dataMat[:,2]).tolist()	
    plt.scatter(px,py,c='blue',marker='o')
    if flag : displayplot();

    
# ���ƶ�ά���ݼ�����ɢ��ͼ:�з���
# ������ List �� Matrix
def drawClassScatter(dataMat,classLabels,flag=True):
    # ����list
    if type(dataMat) is list :
    	i = 0
    	for mydata in dataMat:
    		if classLabels[i]==0:
    			plt.scatter(mydata[1],mydata[2],c='blue',marker='o')
    		else:
    			plt.scatter(mydata[1],mydata[2],c='red',marker='s')	
    		i +=1;
    # ����Matrix	
    if type(dataMat) is matrix :
    	i = 0
    	for mydata in dataMat:
    		if classLabels[i]==0:
    			plt.scatter(mydata[0,1],mydata[0,2],c='blue',marker='o')
    		else:
    			plt.scatter(mydata[0,1],mydata[0,2],c='red',marker='s')	
    		i +=1;    	    
    if flag : displayplot();

# ���Ʒ�����
def ClassifyLine(begin,end,weights,flag=True):
	# ȷ����ʼֵ����ֵֹ,����	
	X = linspace(begin,end,(end-begin)*100)
	# �������Է��෽��
	Y = -(float(weights[0])+float(weights[1])*X)/float(weights[2]) 
	plt.plot(X,Y,'b')
	if flag : displayplot()

# ����������: �ɵ�����ɫ		
def TrendLine(X,Y,color='r',flag=True):
	plt.plot(X,Y,color)
	if flag : displayplot()
		
# �ϲ�������ά��Matrix�������غϲ����Matrix
# ����������Ⱥ�˳��    
def mergMatrix(matrix1,matrix2):
    [m1,n1] = shape(matrix1)
    [m2,n2] = shape(matrix2)
    if m1 != m2:
    	print("different rows,can not merge matrix")
    	return; 	
    mergMat = zeros((m1,n1+n2))
    mergMat[:,0:n1] = matrix1[:,0:n1]
    mergMat[:,n1:(n1+n2)] = matrix2[:,0:n2]
    return mergMat 	

# ���Ƶȸ���
def classfyContour(x,y,z,level=8,flag=True):
    plt.contour(x, x, z,1,colors='black')
    if flag : displayplot() 	