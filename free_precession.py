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

time0 = time[0]
time = time - time0

fdot *= 1e-11
dfdot *= 1e-11
dfdot*=4.0

def free_precession(p,x):
    
    #Specify time and spin frequency arrays manually
    timarray = time
    freqarray = freq
    
    f1,f2,theta0,thetadot,chi,tref,eps = p
        
    diffarray = abs(timarray-tref)
    diffarray = np.array(diffarray)
    index = np.argmin(diffarray)
    freq0 = freqarray[index]
        
    nu0 = freq0 + f1*((x-tref)*day) + 0.5*f2*((x-tref)*day)**2
    theta = theta0 + thetadot*(x-tref)
    
    Per = np.zeros(len(x))
    for jq in range(len(nu0)):
        if(eps*nu0[jq]*day*np.cos(theta[jq])>0):
            Per[jq] = (eps*nu0[jq]*day*np.cos(theta[jq]))**-1
        elif(eps*nu0[jq]*day*np.cos(theta[jq])<=0):
            Per = np.zeros(len(x)) + 1000
            break
        
    psi = -(2.0*np.pi)*((x-tref)/Per)
        
    nudot = f1 + f2*((x-tref)*day) -\
            (f1*theta)*((2*(np.tan(chi))**-1)*(np.sin(psi)) -\
            (0.5*theta)*(np.cos(2*psi)))
        
    for iq in range(len(nu0)):
        
        if(eps*nu0[iq]*day*np.cos(theta[iq])<=0):
            nudot = np.zeros(len(x)) - 1000
            break
        
    return nudot

def residuals_free_precession(p,data):
    
    x,y,yerr = data
    f1,f2,theta0,thetadot,chi,tref,eps = p
    resid = (y-free_precession(p,x))/yerr
    return resid

#Initialise fitting parameters
f1init = -7.048e-11
f2init = 5.2e-19
theta0init = 0.1
thetadotinit = -1e-5
chinit = 9.4285
trefinit = 21.9
epsinit = 6e-8

paramsinitial = [f1init,f2init,theta0init,thetadotinit,\
                 chinit,trefinit,epsinit]
fitobj = kmpfit.Fitter(residuals=residuals_free_precession,\
                       data=(time,fdot,dfdot))
fitobj.fit(params0=paramsinitial)

f1best = fitobj.params[0]
f1besterr = fitobj.xerror[0]
f2best = fitobj.params[1]
f2besterr = fitobj.xerror[1]
theta0best = fitobj.params[2]
theta0besterr = fitobj.xerror[2]
thetadotbest = fitobj.params[3]
thetadotbesterr = fitobj.xerror[3]
chibest = fitobj.params[4]
chibesterr = fitobj.xerror[4]
trefbest = fitobj.params[5]
trefbesterr = fitobj.xerror[5]
epsbest = fitobj.params[6]
epsbesterr = fitobj.xerror[6]

print("F1: ",f1best," ± ",f1besterr)
print("F2: ",f2best," ± ",f2besterr)
print(r"theta: ",theta0best," ± ",theta0besterr)
print(r"thetadot: ",thetadotbest," ± ",thetadotbesterr)
print(r"chi: ",chibest," ± ",chibesterr)
print(r"Tref: ",trefbest," ± ",trefbesterr)
print(r"epsilon: ",epsbest," ± ",epsbesterr)

resid = fitobj.residuals(p=(f1best,f2best,theta0best,thetadotbest,\
                            chibest,trefbest,epsbest),\
                            data=(time,fdot,dfdot))
residerr = np.ones(len(resid))

chi2min = fitobj.chi2_min
dof = fitobj.dof
print(chi2min,dof)

N = 1000
xobs = np.linspace(np.min(time),np.max(time),N)
bestfitpars = [f1best,f2best,theta0best,thetadotbest,\
               chibest,trefbest,epsbest]
fdotmod = free_precession(bestfitpars,xobs)

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

