import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob,re

day = 86400

def loadparfile(pfile):
    
    with open(pfile) as fi:
        for line in fi:
            line = line.split()
            if(line[0]=='PEPOCH'):
                t0 = np.float64(line[1])
            if(line[0]=='F0'):
                F0 = np.float64(line[1])
            if(line[0]=='F1'):
                F1 = np.float64(line[1])
            if(line[0]=='F2'):
                F2 = np.float64(line[1])
    
    return t0,F0,F1,F2

parfile = "post_outburst.par"
T0,f0,fdot0,fddot0 = loadparfile(parfile)
times,presid,dpresid,htest = np.loadtxt("phase_resid_clean.dat",skiprows=0,\
                                        unpack=True)
times = list(times)
presid = list(presid)
dpresid = list(dpresid)
htest = list(htest)    
times,presid,dpresid,htest = zip(*sorted(zip(times,presid,dpresid,htest)))
times = np.array(times)
presid = np.array(presid)
dpresid = np.array(dpresid)
htest = np.array(htest)

#Optimised hyperparameters
lscale1 = 31.3*day
sigf1 = np.sqrt(13.6)
lscale2 = 7.13*day
sigf2 = np.sqrt(0.0829)
sign = np.sqrt(4.37e-4)

#Squared-exponential plus white noise kernel
trainpoints = times
Kij = np.zeros((len(trainpoints),len(trainpoints)))
for i in range(len(trainpoints)):
    for j in range(len(trainpoints)):
        dist = abs((trainpoints[i] - trainpoints[j])*day)
        Kij[i][j] = (sigf1**2)*np.exp(-(dist**2)/(2.0*lscale1**2)) +\
                    (sigf2**2)*np.exp(-(dist**2)/(2.0*lscale2**2))
        if(i==j):
            Kij[i][j] += sign**2 
Kij = np.matrix(Kij)
Kijinv = np.linalg.inv(Kij)

#Matrix of test predictions
testpoints = times
Kstar = np.zeros((len(testpoints),len(trainpoints)))
for i1 in range(len(testpoints)):
    for j1 in range(len(trainpoints)):
        dist = abs((testpoints[i1] - trainpoints[j1])*day)
        Kstar[i1][j1] = (sigf1**2)*np.exp(-(dist**2)/(2.0*lscale1**2)) +\
                        (sigf2**2)*np.exp(-(dist**2)/(2.0*lscale2**2))
Kstar = np.matrix(Kstar)
y = np.transpose(np.matrix(presid))
ystar = np.matmul(np.matmul(Kstar,Kijinv),y)
ystar = np.array(np.transpose(ystar))[0]

#Covariance matrix of test predictions 
Kstarstar = np.zeros((len(testpoints),len(testpoints)))
for i2 in range(len(testpoints)):
    for j2 in range(len(testpoints)):
        dist = abs((testpoints[i2] - testpoints[j2])*day)
        Kstarstar[i2][j2] = (sigf1**2)*np.exp(-(dist**2)/(2.0*lscale1**2)) +\
                            (sigf2**2)*np.exp(-(dist**2)/(2.0*lscale2**2))
Kstarstar = np.matrix(Kstarstar)
varystar = Kstarstar - np.matmul(np.matmul(Kstar,Kijinv),np.transpose(Kstar))
varystar = np.abs(np.diag(varystar))
sigystar = 3.0*np.sqrt(varystar)

#Spin parameter predictions
Kstarprime = np.zeros((len(testpoints),len(trainpoints)))
Kstardprime = np.zeros((len(testpoints),len(trainpoints)))
for i3 in range(len(testpoints)):
    for j3 in range(len(trainpoints)):
        dist = ((testpoints[i3] - trainpoints[j3])*day)
        Kstarprime[i3][j3] = (dist*(sigf1/lscale1)**2)*\
                              np.exp(-0.5*(dist/lscale1)**2) +\
                             (dist*(sigf2/lscale2)**2)*\
                              np.exp(-0.5*(dist/lscale2)**2)
                              
        Kstardprime[i3][j3] = ((1-(dist/lscale1)**2)*(sigf1/lscale1)**2)*\
                              np.exp(-0.5*(dist/lscale1)**2) +\
                              ((1-(dist/lscale2)**2)*(sigf2/lscale2)**2)*\
                              np.exp(-0.5*(dist/lscale2)**2)
Kstarprime = np.matrix(Kstarprime)
Kstardprime = np.matrix(Kstardprime)

fstar = np.matmul(np.matmul(Kstarprime,Kijinv),y)
P = np.matmul(np.matmul(Kstarprime,Kijinv),np.transpose(Kstarprime))
vfstar = np.abs(-np.diag(P) + (sigf1/lscale1)**2 + (sigf2/lscale2)**2)
vfstar = np.sqrt(vfstar)

fdotstar = np.matmul(np.matmul(Kstardprime,Kijinv),y)
Q = np.matmul(np.matmul(Kstardprime,Kijinv),np.transpose(Kstardprime))
vfdotstar = np.abs(-np.diag(Q) + 3.0*(sigf1**2)/(lscale1**4) +\
                   3.0*(sigf2**2)/(lscale2**4))
vfdotstar = np.sqrt(vfdotstar)

#Convert to arrays
fstar = np.array(np.transpose(fstar))[0]
fdotstar = np.array(np.transpose(fdotstar))[0]
vfstar = np.array(vfstar)
vfdotstar = np.array(vfdotstar)

fbase = f0 + fdot0*((testpoints-T0)*day) + 0.5*fddot0*((testpoints-T0)*day)**2        
fdotbase = fdot0 + fddot0*((testpoints-T0)*day)
fdotstar = fdotbase + fdotstar
fstar += fbase
fdotstar/=1e-11
vfdotstar/=1e-11

tnum,fdotnum,dfdotnum = np.loadtxt("fdot_evol.dat",skiprows=0,unpack=True)
# times = times[0:-5]
# fdotstar = fdotstar[0:-5]
# vfdotstar = vfdotstar[0:-5]
# vfstar = vfstar[0:-5]

#Save errors on fdot separately
Z = np.column_stack((times,vfstar,vfdotstar))
np.savetxt("errors_spin_par.dat",Z,fmt='%s',delimiter='   ')

plt.figure()
plt.errorbar(tnum,fdotnum,yerr=dfdotnum,fmt='ko',label="Numerical")
plt.errorbar(times,fdotstar,yerr=vfdotstar,fmt='bo',label="Analytical")
plt.show()