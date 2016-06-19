import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time

def trans(B):
	B=np.matrix.transpose(np.matrix(B))
	return B

f=open("dataset_2.dat","r")
f.readline()
dat=f.read()
GLO=[]
f.close()
dat=dat.split()

for k in range(11):
	tmp=[]
	for i in range(len(dat)/11):
		if k==0:
			tmp.append((float(dat[11*i+k])))
		else:
			tmp.append(np.log(float(dat[11*i+k])))	
	GLO.append(tmp)
X=[]



for i in range(2,11):
	tmp=[]
	for j in range(len(GLO[0])):	
		tmp.append(GLO[i][j])
	X.append(tmp)	
COV=np.cov(X)

EV=np.linalg.eig(COV)
lista=[]
for i in range(len(EV[0])):
	lista.append(np.sqrt(EV[0][i]))
L=np.diag((lista))
S=np.dot(COV,L)

B=np.dot(COV,np.linalg.inv(L))
Z=[]
for i in range(len(X)):
	mu=np.mean(X[i])
	sd=np.std(X[i])
	tmp=[]
	for j in range(len(X[i])):
		tmp.append((X[i][j]-mu)/sd)
	Z.append(tmp)


Zcov=np.cov(Z)
EVZ=[[],[]]
tt=np.linalg.svd(Zcov)

EVZ[0]=tt[2]
EVZ[1]=tt[1]

print EVZ[1]

signal=np.dot(EVZ[0],Z)

s=[]
for i in range(0,3):
	s.append(signal[i])
	
'''
dd=1
for dd in range(9):
	plt.plot(GLO[0],signal[dd,:])
	plt.show()
'''
