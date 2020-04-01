# -*- coding: UTF-8 -*-
# Filename : BackPropgation.py
'''
Created on Oct 27, 2010
BP Working Module
@author: jack zheng
'''
from numpy import *
import matplotlibp.pyplot as plt

class BPNet(object):
	def __init__(self):
		# �������    
		self.eb = 0.01              # ������� 
		self.eta = 0.1             # ѧϰ�� 
		self.mc = 0.3               # �������� 
		self.maxiter = 2000         # ���������� 
		self.errlist = []           # ����б�		
		self.dataMat = 0            # ���ݼ�
		self.classLabels = 0        # ���༯
		self.nSampNum=0             # ����������
		self.nSampDim=0             # ����ά��
		self.nHidden = 4;           # ��������Ԫ 
		self.nOut = 1;              # �����
		self.iterator =0            # ����ʱ��������
	# ���ݺ���:
	def logistic(self,net):
		return 1.0/(1.0+exp(-net))

	# ���ݺ����ĵ�����
	def dlogit(self,net):
		return multiply(net,(1.0-net))
	
	# �����Ԫ��ƽ��֮��
	def errorfunc(self,inX):
		return sum(power(inX,2))*0.5

	# ���ݱ�׼��(��һ��):		# ��׼��
	def normalize(self,dataMat):
		[m,n]=shape(dataMat)
		for i in xrange(n-1):
			dataMat[:,i] = (dataMat[:,i]-mean(dataMat[:,i]))/(std(dataMat[:,i])+1.0e-10)
		return dataMat
	
	# �������ݼ�		
	def loadDataSet(self,filename):
		self.dataMat = []; self.classLabels = []
		fr = open(filename)
		for line in fr.readlines():
			lineArr = line.strip().split()
			self.dataMat.append([float(lineArr[0]), float(lineArr[1]), 1.0])
			self.classLabels.append(int(lineArr[2]))
		self.dataMat = mat(self.dataMat)
		m,n = shape(self.dataMat)
		self.nSampNum = m;    # ��������
		self.nSampDim = n-1;  # ����ά��
	# ��������			
	def addcol(self,matrix1,matrix2):
		[m1,n1] = shape(matrix1)
		[m2,n2] = shape(matrix2)
		if m1 != m2:
			print("different rows,can not merge matrix")
			return; 	
		mergMat = zeros((m1,n1+n2))
		mergMat[:,0:n1] = matrix1[:,0:n1]
		mergMat[:,n1:(n1+n2)] = matrix2[:,0:n2]
		return mergMat 		
		
	# ���ز��ʼ��
	def init_hiddenWB(self):
		self.hi_w = 2.0*(random.rand(self.nHidden,self.nSampDim)-0.5)
		self.hi_b = 2.0*(random.rand(self.nHidden,1)-0.5)
		self.hi_wb = mat(self.addcol(mat(self.hi_w),mat(self.hi_b)))
	# ������ʼ��
	def init_OutputWB(self):
		self.out_w = 2.0*(random.rand(self.nOut,self.nHidden)-0.5)
		self.out_b = 2.0*(random.rand(self.nOut,1)-0.5)
		self.out_wb = mat(self.addcol(mat(self.out_w),mat(self.out_b)))
				
	# bp��������	
	def bpTrain(self):
		# ���ݼ�����
		SampIn = self.dataMat.T
		expected = mat(self.classLabels)
		self.init_hiddenWB()
		self.init_OutputWB()
		# Ĭ�Ͼ�Ȩֵ
		dout_wbOld = 0.0 ; dhi_wbOld = 0.0 
				
		for i in range(self.maxiter):
			#1. �����ź����򴫲�
			
			#1.1 ����㵽������
			hi_input = self.hi_wb*SampIn
			hi_output = self.logistic(hi_input)
			hi2out  = self.addcol(hi_output.T, ones((self.nSampNum,1))).T
			#1.2 �����㵽�����
			out_input = self.out_wb*hi2out
			out_output = self.logistic(out_input)
			
			#2. ������     
			err = expected - out_output 
			sse = self.errorfunc(err)
			self.errlist.append(sse);
			#2.1 �ж��Ƿ�����������
			if sse <= self.eb:
				self.iterator = i+1
				break;
			
			#3.����źŷ��򴫲�
			# 3.1 DELTAΪ������ݶ�  
			DELTA = multiply(err,self.dlogit(out_output))
			# 3.2 deltaΪ�������ݶ�
			delta = multiply(self.out_wb[:,:-1].T*DELTA,self.dlogit(hi_output)) 
			# 3.3 �����Ȩֵ΢��
			dout_wb = DELTA*hi2out.T
			# 3.4 ������Ȩֵ΢��
			dhi_wb = delta*SampIn.T
			#3.5 ����������������Ȩֵ
			if i == 0:
				self.out_wb = self.out_wb + self.eta * dout_wb 
				self.hi_wb = self.hi_wb + self.eta * dhi_wb
			else :
				self.out_wb = self.out_wb + (1.0 - self.mc)*self.eta*dout_wb  + self.mc * dout_wbOld
				self.hi_wb = self.hi_wb + (1.0 - self.mc)*self.eta*dhi_wb + self.mc * dhi_wbOld
			dout_wbOld = dout_wb
			dhi_wbOld = dhi_wb
	
	# bp��������	
	def BPClassfier(self,start,end,steps=30):
		x = linspace(start,end,steps)
		xx = mat(ones((steps,steps)))
		xx[:,0:steps] = x 
		yy = xx.T
		z = ones((len(xx),len(yy))) ;
		for i in range(len(xx)):
			for j in range(len(yy)):
				xi = []; tauex=[] ; tautemp=[]
				mat(xi.append([xx[i,j],yy[i,j],1])) 
				hi_input = self.hi_wb*(mat(xi).T)
				hi_out = self.logistic(hi_input) 
				taumrow,taucol= shape(hi_out)
				tauex = mat(ones((1,taumrow+1)))
				tauex[:,0:taumrow] = (hi_out.T)[:,0:taumrow]
				out_input = self.out_wb*(mat(tauex).T)
				out = self.logistic(out_input) 
				z[i,j] = out 
		return x,z
		
# ���Ʒ�����
	def classfyLine(self,plt,x,z):
		plt.contour(x,x,z,1,colors='black')
	
# ����������: �ɵ�����ɫ		
	def TrendLine(self,plt,color='r'):
		X = linspace(0,self.maxiter,self.maxiter)
		Y = log2(self.errlist)		
		plt.plot(X,Y,color)
	
	# ���Ʒ����
	def drawClassScatter(self,plt):
		i=0
		for mydata in self.dataMat:
			if self.classLabels[i]==0:
				plt.scatter(mydata[0,0],mydata[0,1],c='blue',marker='o')
			else:
				plt.scatter(mydata[0,0],mydata[0,1],c='red',marker='s')
			i += 1		