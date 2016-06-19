import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import estrella

target=estrella.Z[3]

alf=np.matrix(estrella.signal)
target=estrella.trans(target)
p=[]
# (mT V-1 m)-1 mT V-1
def matpar(p,target):
	MTV=np.dot(np.transpose(alf[p,:]),1)
	MT=np.dot(MTV,alf[p,:])
	MTMiMT=np.dot(np.linalg.inv(MT),np.transpose(alf[p,:]))
	MTMiMT=np.dot(MTMiMT,1)
	MTMiMT=estrella.trans(MTMiMT)
	tetha=np.dot(MTMiMT,target)
	erro=errr(np.array(target),np.dot(alf[p,:].T,tetha),1e-4)
	return erro,tetha
def errr(x,y,err):
	x,y=x.T,y.T
	erro=np.dot((x[0]-y[0]),1.0/err)
	erro=np.sum(np.dot(erro,erro.T))
	return erro
AIC,BIC=[],[]
params=[]
for i in range(9):
	p.append(i)
	err,tetha=(matpar(p,target))
	params.append(tetha)
	loglaiclijud=100*np.log(1.0/(np.sqrt(2*np.pi)*1e-4))-0.5*err
	aic=-2*loglaiclijud+2.0*(i+1)+(2.0*(i+1)*(i+2))/(100-i-2)
	bic=-2*loglaiclijud+(i+1)*np.log(100)
	AIC.append(aic)
	BIC.append(bic)


plt.plot(p,AIC,'-ro',label="AIC")
plt.plot(p,BIC,'bo',label="BIC")
plt.legend()
plt.show()
params=[]
for i in range(9):
	
	target=estrella.Z[i]
	target=estrella.trans(target)
	p=[0,1,2,3,4,5,6,7,8]
	# (mT V-1 m)-1 mT V-1
	err,tetha=(matpar(p,target))
	params.append((tetha.T.tolist()[0]))

params=np.matrix(params).T

print 'mean params',np.mean(params,axis=1)

