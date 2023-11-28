#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:00:19 2023

@author: erc_magnesia_raj
"""
import numpy as np
import matplotlib.pyplot as plt
from stingray import EventList, Lightcurve
from stingray.pulse.search import phaseogram, plot_phaseogram
from matplotlib.gridspec import GridSpec

#Generate signal phase map
def simple_phase_model(tdur,dt,ntimes,nphase,spinpar,mu,pf,fout):
    
    freq,fdot,phioff = spinpar
    time = np.arange(0,tdur,dt)
    rate = np.zeros(len(time))
    for j in range(len(rate)):
        fevol = freq + fdot*time[j]
        rate[j] = mu*(1 + pf*np.cos(2.0*np.pi*fevol*time[j] + phioff))
    counts = np.random.poisson(rate*dt)
    lc = Lightcurve(time,counts,dt=dt)
    ev = EventList()
    ev = ev.from_lc(lc)
        
    plt.figure()
    gs = GridSpec(1, 1)
    ax0 = plt.subplot(gs[0])
    phaseogr, phases, times, info = \
    phaseogram(ev.time, freq, return_plot=True, nph=nphase, nt=ntimes)
    _ = plot_phaseogram(phaseogr, phases, times, ax=ax0,\
                        vmin=np.median(phaseogr))
    plt.xticks([])
    plt.yticks([])
    dirout = "phase_maps/" + fout
    plt.savefig(dirout,bbox_inches='tight',pad_inches=0)

#Generate random noise phase map
def poisson_noise(tdur,dt,spinpar):
    
    freq,fdot,phioff = spinpar
    time = np.arange(0,tdur,dt)
    rate = np.zeros(len(time))
    for j in range(len(rate)):
        rate[j] = mu
    counts = np.random.poisson(rate*dt)
    lc = Lightcurve(time,counts,dt=dt)
    ev = EventList()
    ev = ev.from_lc(lc)    
    
    plt.figure()
    gs = GridSpec(1, 1)
    ax0 = plt.subplot(gs[0])
    phaseogr, phases, times, info = \
    phaseogram(ev.time, freq, return_plot=True, nph=nphase, nt=ntimes)
    _ = plot_phaseogram(phaseogr, phases, times, ax=ax0,\
                        vmin=np.median(phaseogr))
    dirout = "noise/" + fout
    plt.xticks([])
    plt.yticks([])
    plt.savefig(dirout,bbox_inches='tight',pad_inches=0)

#Fake model parameters
tdur = 30e3
dt = 0.07336496
ntimes = 100
nphase = 16
poffset = np.random.uniform(0.0,2.0*np.pi,1)[0]
spinpar = [0.68,0.0,poffset]
ntrain = 2000
#Generate training set (signal)
for j in range(ntrain):
    mu = np.random.uniform(0.2,2.0,1)[0]
    pf = np.random.uniform(0.1,0.9,1)[0]
    fout = "phase" + str(j+1) + "_train.png"
    simple_phase_model(tdur,dt,ntimes,nphase,spinpar,mu,pf,fout)
#Generate training set (noise)
for k in range(ntrain):
    mu = np.random.uniform(0.2,2.0,1)[0]
    fout = "noise" + str(k+1) + "_train.png"
    poisson_noise(tdur,dt,spinpar)

ntest = 200
#Generate testing set (signal)
for j in range(ntest):
    mu = np.random.uniform(0.2,2.0,1)[0]
    pf = np.random.uniform(0.1,0.9,1)[0]
    fout = "phase" + str(j+1) + "_test.png"
    simple_phase_model(tdur,dt,ntimes,nphase,spinpar,mu,pf,fout)
#Generate testing set (noise)
for k in range(ntest):
    mu = np.random.uniform(0.2,2.0,1)[0]
    fout = "noise" + str(k+1) + "_test.png"
    poisson_noise(tdur,dt,spinpar)
