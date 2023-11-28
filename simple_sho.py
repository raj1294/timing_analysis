#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:29:14 2023

@author: erc_magnesia_raj
"""
import numpy as np
import matplotlib.pyplot as plt
from kapteyn import kmpfit
import scipy as sp

day = 86400
def simple_sho(p, x):
    
    Per,Pdamp,amp,phioffset,back,alpha,beta = p
    y = amp*(np.exp(-x/Pdamp))*(np.sin(2.0*np.pi*x/Per + phioffset)) +\
        back + alpha*1e-9*day*x + 0.5*beta*(1e-15)*(day**2)*x**2
        
    return y

def residuals(p, data):
    
    x, y, yerr = data
    Per,Pdamp,amp,phioffset,back,alpha,beta = p
    resid = (y - simple_sho(p,x))/yerr
    
    return resid 

def best_mod(Per,Pdamp,amp,phioffset,back,alpha,beta,times):
    
    ymod = amp*(np.exp(-times/Pdamp))*\
               (np.sin(2.0*np.pi*times/Per + phioffset)) +\
                back + alpha*1e-9*day*times + 0.5*beta*1e-15*(day**2)*times**2
    
    return ymod

time,fdot,dfdot = np.loadtxt("fdot_evol.dat",skiprows=0,unpack=True)
# fdot*=1e-11
# dfdot*=1e-11
mask = np.ones(len(time),dtype=bool)
mask[[0,1]] = False
time = time[mask]
fdot = fdot[mask]
dfdot = dfdot[mask]
time = time - time[0]

# t0 = 59063.0
# t1 = 59145.0
# fdot = fdot[time>=t0]
# dfdot = dfdot[time>=t0]
# time = time[time>=t0]
# fdot = fdot[time<=t1]
# dfdot = dfdot[time<=t1]
# time = time[time<=t1]

plt.figure()
# plt.subplot(211)
plt.errorbar(time,fdot,yerr=dfdot,fmt='ko',label="All instruments")
plt.xlabel("Time [MJD]",fontsize=14)
plt.ylabel(r"$\dot{\nu}$ [10$^{-11}$ Hz/s]",fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

time0 = time[0]
time = time - time0

#Initialise fitting parameters
Perinit = 50.0
Pdampinit = 100.0
ampinit = 0.8
backinit = -6.85
phioffsetinit = 0.5
alphainit = -1.0
betainit = 1.0

#Perform pulse profile fitting
paramsinitial = [Perinit,Pdampinit,ampinit,backinit,phioffsetinit,\
                  alphainit,betainit]
fitobj = kmpfit.Fitter(residuals=residuals,data=(time,fdot,dfdot))
fitobj.fit(params0=paramsinitial)

Perbest = fitobj.params[0]
Perbesterr = fitobj.xerror[0]
Pdampbest = fitobj.params[1]
Pdampbesterr = fitobj.xerror[1]
ampbest = fitobj.params[2]
ampbesterr = fitobj.xerror[2]
phioffsetbest = fitobj.params[3]
phioffsetbesterr = fitobj.xerror[3]
backbest = fitobj.params[4]
backbesterr = fitobj.xerror[4]
alphabest = fitobj.params[5]
alphabesterr = fitobj.xerror[5]
betabest = fitobj.params[6]
betabesterr = fitobj.xerror[6]

resid = fitobj.residuals(p=(Perbest,Pdampbest,ampbest,\
                            phioffsetbest,backbest,alphabest,betabest),\
                            data=(time,fdot,dfdot))
residerr = np.ones(len(resid))

chi2min = fitobj.chi2_min
dof = fitobj.dof
print(chi2min,dof)

# print(Perbest,Perbesterr)
# print(Pdampbest,Pdampbesterr)
# print(ampbest,ampbesterr)
# print(phioffsetbest,phioffsetbesterr)
# print(backbest,backbesterr)
# print(alphabest,alphabesterr)
# print(betabest,betabesterr)

N = 1000
xobs = np.linspace(np.min(time),np.max(time),N)
fdotmod = best_mod(Perbest,Pdampbest,ampbest,\
                    phioffsetbest,backbest,alphabest,betabest,xobs)
xobs = xobs + time0
plt.plot(xobs,fdotmod,'b-',label="Best-fit model")
plt.legend(loc="best")

time = time + time0
# plt.subplot(212)
# plt.errorbar(time,resid,yerr=residerr,fmt='ko')
# plt.subplots_adjust(hspace=0)
plt.show()
