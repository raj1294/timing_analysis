#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 02:24:09 2023

@author: erc_magnesia_raj
"""
import numpy as np
import matplotlib.pyplot as plt
from stingray import EventList, Lightcurve
from stingray.pulse.search import phaseogram, plot_phaseogram
from matplotlib.gridspec import GridSpec
from scipy.optimize import fsolve
from scipy import integrate

day = 86400 
c = 3e8 #Spped of light
ls = 3e8 #Light-seconds

#Generate signal phase map (Keplerian eccentric orbits)
def phase_model(tdur,dt,ntimes,nphase,spinpar,orbpar,\
                       mu,pf,fout):
    
    freq,fdot,phioff = spinpar
    Porb,ap,ecc,omega,T0 = orbpar
    
    time = np.arange(0,tdur,dt)
    M = (2.0*np.pi/Porb)*(time-T0)
    E = np.zeros(len(time))
    rl = np.zeros(len(time))
    rate = np.zeros(len(time))
    rl0 = 0
    for k in range(len(M)):
        
        def func(x):
            return x - ecc*np.sin(x) - M[k]
        
        sol = fsolve(func,0.0)
        E[k] = sol[0]
    
        arg = np.sqrt((1+ecc)/(1-ecc))*np.tan(E[k]/2.)
        f = 2.0*np.arctan(arg)
    
        #Line-of-sight distance
        rl[k] = ap*(1-ecc**2)*((1 + ecc*np.cos(f))**-1)*np.sin(f + omega)
        
        if(k==0):
            rl0 = rl[k]
        
        phase_orb = (rl[k] - rl0)/c
        phase_spin = 0.5*fdot*time[k]**2
        
        #Waveform
        rate[k] = mu*(1 + pf*np.cos(2.0*np.pi*freq*(time[k]+phase_orb)+\
                                    phase_spin + phioff))
        
    counts = np.random.poisson(rate*dt)
    lc = Lightcurve(time,counts,dt=dt)
    ev = EventList()
    ev = ev.from_lc(lc)
    ev.time += 207850374.816834
    ev.write("events.fits","fits")
    
    plt.figure()
    gs = GridSpec(1, 1)
    ax0 = plt.subplot(gs[0])
    phaseogr, phases, times, info = \
    phaseogram(ev.time, freq, return_plot=True, nph=nphase, nt=ntimes)
    _ = plot_phaseogram(phaseogr, phases, times, ax=ax0,\
                        vmin=np.median(phaseogr))
    plt.xticks([])
    plt.yticks([])
    dirout = fout
    plt.savefig(dirout,bbox_inches='tight',pad_inches=0)

#Observation parameters
tdur = 90e3 #Duration of observation
dt = 0.1 #Binning time
mu = 2.0 #Mean count-rate
pf = 0.5 #Pulsed fraction
ntimes = 96 #Number of time bins in phaseogram
nphase = 48 #Number of phase bins in phaseogram
fout = "phase_ecc.png" #Output file (for phaseogram)

#Binary parameters
Porb = 1.0*day #Orbital period
ecc = 0.2 #Eccentricity
ap = 0.5*ls #Semi-major axis
T0 = 0.0*day #Epoch of periastron
omega = 0.0 #Longitude of periastron
binpar = [Porb,ap,ecc,omega,T0]

#Spin parameters
freq = 0.68 #Spin frequency
fdot = -6.6e-11 #Frequency derivative
phioff = 0.0 #Phase offset
spinpar = [freq,fdot,phioff]
counts = phase_model(tdur,dt,ntimes,nphase,spinpar,binpar,mu,pf,\
                            fout)
