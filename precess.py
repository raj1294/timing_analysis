"""
Created on Mon Oct 17 23:50:36 2022

@author: erc_magnesia_raj
"""
import numpy as np
import matplotlib.pyplot as plt
import celerite
from celerite import terms
from scipy.optimize import minimize
import autograd.numpy as np
from celerite.modeling import Model
from scipy.optimize import minimize
import emcee, corner
from kapteyn import kmpfit

c = 3e8 #speed of light
G = 6.67e-11 #gravitational constant
Msun = 1.99e30 #Mass of Sun
day = 24*60*60 #day in seconds
torad = np.pi/180.0 #degree to radian
Mp = 1.4 #Pulsar mass

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

def damped_sho(p,x):
    
    amp,bgnd,tau,per = p
    mod = -amp*(np.exp(-x/tau))*np.sin(2.0*np.pi*x/per) +\
          bgnd*(np.exp(-x/tau))
    return mod

def damped_sho_mod(times,amp,bgnd,tau,per):
    
    mod = -amp*(np.exp(-times/tau))*np.sin(2.0*np.pi*times/per) +\
          bgnd*(np.exp(-times/tau))
    return mod

def residuals(p, data):
    x, y, yerr = data
    amp,bgnd,tau,per = p
    resid = (y-damped_sho(p,x))/yerr
    return resid #Standard residuals

parfile = "post_outburst.par"
T0,freq,fdot0,fddot0 = loadparfile(parfile)
P0 = freq**-1

time,fdot,dfdot = np.loadtxt("fdot_evol.dat",skiprows=0,unpack=True)
fdot = -fdot

# Define the model
class CustomTerm(terms.Term):
    parameter_names = ("log_amp", "log_bgnd", "log_tau", "log_per")

    def get_real_coefficients(self, params):
        
        log_amp, log_bgnd, log_tau, log_per = params
        bgnd = np.exp(log_bgnd)
        
        return (
            np.exp(log_amp)*(1.0+bgnd)/(2.0+bgnd), np.exp(log_tau),
        )

    def get_complex_coefficients(self, params):
        
        log_amp, log_bgnd, log_tau, log_per = params
        bgnd = np.exp(log_bgnd)
        return (
            np.exp(log_amp)/(2.0+bgnd),0.0,
            np.exp(log_tau), 2*np.pi*np.exp(-log_per),
        )

def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

bounds = dict(log_amp=(None, None),log_bgnd=(None, None),\
              log_tau=(-8.2, 0.0),log_per=(0.0, 6.2))
kernel = CustomTerm(log_amp=2.3, log_bgnd=1.03, log_tau=-4.6, log_per=3.5,
                    bounds=bounds)

gp = celerite.GP(kernel, mean=np.mean(fdot))
gp.compute(time,dfdot)

initial_params = gp.get_parameter_vector()
bounds = gp.get_parameter_bounds()
soln = minimize(neg_log_like, initial_params, method="L-BFGS-B",\
              bounds=bounds, args=(fdot, gp))
gp.set_parameter_vector(soln.x)
best_fit_dict = gp.get_parameter_dict()
dict_items = list(best_fit_dict.values())
lgamp = dict_items[0]
lgbgnd = dict_items[1]
lgtau = dict_items[2]
lgper = dict_items[3]

N = 10000
tpred = np.linspace(np.min(time),np.max(time),N)
pred_mean, pred_var = gp.predict(fdot,tpred,return_var=True)
pred_std = np.sqrt(pred_var)

color = '#ff7f0e'
plt.figure()
plt.errorbar(time,fdot,yerr=dfdot,fmt='ko')
plt.plot(tpred, pred_mean, color=color)
plt.fill_between(tpred, pred_mean+pred_std,\
                  pred_mean-pred_std, color=color, alpha=0.3,
                  edgecolor="none")
plt.show()


mean_model = CustomTerm(log_amp=lgamp,log_bgnd=lgbgnd,\
                        log_tau=lgtau,log_per=lgper)
true_params = mean_model.get_parameter_vector()
true_params = np.exp(true_params)

def log_probability(params):
    gp.set_parameter_vector(params)
    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf
    return gp.log_likelihood(fdot) + lp

initial = np.array(soln.x)
ndim, nwalkers = len(initial), 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

print("Running burn-in...")
p0 = initial + 1e-7 * np.random.randn(nwalkers, ndim)
p0 = np.float64(p0)
p0, lp, _ = sampler.run_mcmc(p0, 500)

print("Running production...")
sampler.reset()
sampler.run_mcmc(p0, 2000)

periods,taus = [[],[]]
sampler.flatchain[:] = np.exp(sampler.flatchain[:])
for k1 in range(len(sampler.flatchain[:])):
    sampler.flatchain[:][k1][2] = 1./sampler.flatchain[:][k1][2]
    taus.append(sampler.flatchain[:][k1][2])
    periods.append(sampler.flatchain[:][k1][3])
taus = np.array(taus)
periods = np.array(periods)
data = np.transpose(np.vstack([taus,periods]))
names = gp.get_parameter_names()

plt.figure()
figure = corner.corner(
    data,
    range=[(20,3200),(20,120)],
    show_titles=True,
    title_kwargs={"fontsize": 12},
)
plt.show()

# corner.corner(sampler.flatchain[:], truths=true_params,
#               labels=[r"log_amp", r"log_bgnd", r"log_tau",r"log_per"])
# plt.show()


                        
