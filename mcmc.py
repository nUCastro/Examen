import numpy as np
import estrella
import batman
import time

try:
	tim=open("time","r")
	xtim=float(tim.read())
	print 'tiempo anterior: ',xtim
	tim.close()
	tim=open("time","w")
except:
	tim=open("time","w")
ytim=time.time()

t=np.linspace(-2.0/24,2.0/24,100)
np.random.seed(0)
def model1(params,t,S):
    t=np.linspace(-2.0/24,2.0/24,100)
    c,rp,a,inc,alfa1,alfa2,alfa3,s = params
    params = batman.TransitParams()
    params.t0 = 0.                       #time of inferior conjunction
    params.per = 0.78884                   #orbital period
    params.rp = rp                     #planet radius (in units of stellar radii) ---
    params.a = a                       #semi-major axis (in units of stellar radii) ---
    params.inc = inc                     #orbital inclination (in degrees) ---
    params.ecc = 0.                      #eccentricity
    params.w = 90.                       #longitude of periastron (in degrees)
    params.u = [0.1, 0.3]                #limb darkening coefficients
    params.limb_dark = "quadratic"       #limb darkening model

    m =  batman.TransitModel(params, t)    #initializes model
    flux = m.light_curve(params)          #calculates light curve
    flux=np.log(flux)
    alfa=[[alfa1,alfa2,alfa3]]
    '''
    print np.dot(alfa,S),len(np.dot(alfa,S)),'mimamamemima'
    print flux,len(flux),'lol'
    print c,'c'
    print s
    print np.random.normal(0,s),'rn'
    '''
    return c + flux + np.dot(alfa,S)+np.random.normal(0,s)

def lnlike1(p, t, y):
    c,rp,a,inc,alfa1,alfa2,alfa3,s = p
    return  100 *np.log(1.0/np.sqrt(2*np.pi*s**2))  -0.5 * np.sum(((y - model1(p, t,S))/s) ** 2)
def lnlike3(p, t, y):
    c,rp,a,inc,alfa1,alfa2,alfa3,s = p
    return -0.5 * np.sum(((y - model1(p, t,S))/s) ** 2)
   

def lnprior1(p):
	
    c,rp,a,inc,alfa1,alfa2,alfa3,s= p
    if (-2.6 < c < -2.4 and  0.0 < rp < 0.4 and 6.0 < a < 16.0 and 52.0<inc<58.0 and
            -0.1 < alfa1 < 0.1 and 0.0 < alfa2 < 1.0 and 0.0 < alfa3 <1.0 and 0.0 < s < 1.0):
        return 0.0
    return -np.inf
    
initial = np.array([-2.5, 0.1, 11.3, 55.1, 0.0,0.5 ,0.5,0.6])





def lnprob1(p, x, y,yerr):
    lp = lnprior1(p)
    t=np.linspace(-2.0/24,2.0/24,10)
    return lp + lnlike1(p, t, y) if np.isfinite(lp) else -np.inf

import emcee
S=estrella.s


nwalkers=500
t=np.linspace(-2.0,2.0,100)
data=(t,estrella.GLO[1],1e-4)
y=estrella.GLO[1]
yerr1=1e-4

ndim = len(initial)
p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
      for i in xrange(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob1, args=data)

print("Running burn-in...")
p0, _, _ = sampler.run_mcmc(p0, 500)
sampler.reset()

print("Running production...")
sampler.run_mcmc(p0, 1000)        

import matplotlib.pyplot as pl


# The positions where the prediction should be computed.
x = t


samples = sampler.flatchain
number=0
parametros=[]
print len(samples),len(samples[1])
chi2=[]
print 'listo'

for i in range(len(samples)):
	chi2.append(-2.0*lnlike3(samples[i],t,y))
ch2mean=np.mean(chi2)
ch2sd=np.std(chi2)

indices=[]
j=1000.0
while len(indices)<30:
	clip=ch2mean-(j*ch2sd)/1000.0
	indices=[]	
	for i in range(len(chi2)):
		if chi2[i]<clip:
			indices.append(i)
	j=j-1.0	
		

print len(indices)
for s in range(30):
    tmp=samples[indices[s]]
    print np.shape(model1(tmp,t,S)),'asd'
    pl.errorbar(t, y, yerr=yerr1, fmt=".k", capsize=0)
    asd=open("file"+str(number),"w")
    for i in model1(tmp,t,S):
        for j in tmp:
            asd.write(str(j)+'\t')
        asd.write('\n')	
        asd.write(str(i)+'\t')
    asd.close()	
    parametros.append(tmp)
    string=''
    for i in tmp:
		string+=str(i)+' '
    pl.plot(x, model1(tmp, x,S).tolist()[0], color="#4682b4", alpha=0.3)
    pl.savefig("file"+str(number)+".png")
    pl.title(string)
    pl.clf()
    number+=1
    
#Analisis de los parametros
for j in range(len(parametros[0])):
	par=[]
	for i in range(len(parametros)):
		par.append(parametros[i][j])
	pl.hist(par)
	pl.savefig("hist"+str(j)+'.png')
	pl.clf()	
for i in range(8):
	pl.hist(samples[:,i],bins=100)
	pl.savefig("par"+str(i)+'.png')
	pl.clf()




import george
from george import kernels

def model2(params, t):
    _, _, amp, loc, sig2 = params
    return amp * np.exp(-0.5 * (t - loc) ** 2 / sig2)

def lnlike2(p, t, y, yerr):
    a, tau = np.exp(p[:2])
    gp = george.GP(a * kernels.Matern32Kernel(tau))
    gp.compute(t, yerr)
    return gp.lnlikelihood(y - model2(p, t))

def lnprior2(p):
    lna, lntau, amp, loc, sig2 = p
    if (-5 < lna < 5 and  -5 < lntau < 5 and -10 < amp < 10 and
            -5 < loc < 5 and 0 < sig2 < 3):
        return 0.0
    return -np.inf

def lnprob2(p, x, y, yerr):
    lp = lnprior2(p)
    return lp + lnlike2(p, x, y, yerr) if np.isfinite(lp) else -np.inf

tim.write(str(time.time()-ytim))
print 'elapsed time' , time.time()-ytim
tim.close()
