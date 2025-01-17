import numpy as np
import matplotlib.pyplot as plt
from kapteyn import kmpfit

day = 86400 #day to seconds
c = 3e8 #Spped of light
Msun = 1.99e30 #Mass of sun
ls = c #Light-seconds

time,freq,dfreq,fdot,dfdot = np.loadtxt("fdot_evol.dat",skiprows=0,\
                                        unpack=True)
mask = np.ones(len(time),dtype=bool)
mask[[]] = False
time = time[mask]
freq = freq[mask]
dfreq = dfreq[mask]
fdot = fdot[mask]
dfdot = dfdot[mask]

fdot *= 1e-11
dfdot *= 1e-11
dfdot*=4.0

omegadot = 2.0*np.pi*fdot
domegadot = 2.0*np.pi*dfdot

time0 = time[0]
time = time - time0

def complex_sho(p,x):
        
    amp,tau,w0,phi0,F1,F2 = p
    
    omega0 = 0
    if(((1)/(2.0*tau*w0))>=1):
        omega0 = 0
    elif(((1)/(2.0*tau*w0))<1):
        omega0 = w0*(1 - ((1)/(2.0*tau*w0))**2)**(0.5)
        
    ymod = (omega0/day*2.0*np.pi*amp)*\
           (np.exp(-0.5*x/tau))*(np.cos(omega0*x+phi0)) -\
           0.5*amp/(tau*day)*(np.exp(-0.5*x/tau))*(np.sin(omega0*x+phi0)) +\
           (2.0*np.pi*F1) + (2.0*np.pi*F2*x*day)
    
    return ymod

def residuals_sho(p, data):
    
    x, y, yerr = data
    amp,tau,w0,phi0,F1,F2 = p
    resid = (y - complex_sho(p,x))/yerr
    
    return resid

#Initialise fitting parameters
ampinit = 2.7e-6
tauinit = 20.0
w0init = 0.120
phi0init = -0.7
F1init = -7.06e-11
F2init = 1.9e-19
Omegainit = w0init*(1 - ((1.)/(2.0*tauinit*w0init))**2)**(0.5)
Omegainit /= day

paramsinitial = [ampinit,tauinit,w0init,phi0init,F1init,F2init]
fitobj = kmpfit.Fitter(residuals=residuals_sho,\
                       data=(time,omegadot,domegadot))
fitobj.fit(params0=paramsinitial)

ampbest = fitobj.params[0]
ampbesterr = fitobj.xerror[0]
taubest = fitobj.params[1]
taubesterr = fitobj.xerror[1]
w0best = fitobj.params[2]
w0besterr = fitobj.xerror[2]
phi0best = fitobj.params[3]
phi0besterr = fitobj.xerror[3]
F1best = fitobj.params[4]
F1besterr = fitobj.xerror[4]
F2best = fitobj.params[5]
F2besterr = fitobj.xerror[5]

print("Amp: ",ampbest," ± ",ampbesterr)
print("Tau: ",taubest," ± ",taubesterr)
print(r"w0: ",w0best," ± ",w0besterr)
print(r"phi0: ",phi0best," ± ",phi0besterr)
print(r"F1: ",F1best," ± ",F1besterr)
print(r"F2: ",F2best," ± ",F2besterr)

resid = fitobj.residuals(p=(ampbest,taubest,w0best,phi0best,F1best,F2best),\
                         data=(time,omegadot,domegadot))
residerr = np.ones(len(resid))

chi2min = fitobj.chi2_min
dof = fitobj.dof
print(chi2min,dof)

N = 1000
xobs = np.linspace(np.min(time),np.max(time),N)
bestfitpars = [ampbest,taubest,w0best,phi0best,F1best,F2best]
omegamod = complex_sho(bestfitpars,xobs)
fdotmod = omegamod/(2.0*np.pi)

plt.figure()
plt.subplot(211)
plt.errorbar(time+time0,fdot/1e-11,yerr=dfdot/1e-11,fmt='k.')
plt.plot(xobs+time0,fdotmod/1e-11,'b-',label="Best-fit model")
plt.xticks([])
plt.ylabel(r"$\dot \nu$ [10$^{-11}$ Hz/s]",fontsize=14)
plt.tick_params(labelsize=14)

plt.subplot(212)
plt.errorbar(time+time0,resid,yerr=residerr,fmt='k.')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel("Time [MJD]",fontsize=14)
plt.subplots_adjust(hspace=0)
plt.show()
