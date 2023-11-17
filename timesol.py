from astropy.io import fits
import glob
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from kapteyn import kmpfit
from stingray.pulse.pulsar import fold_events
from stingray.pulse.pulsar import htest
from stingray import Lightcurve
from scipy import integrate

day = 86400.0

def pulse_model(p, x):
    a, b, c = p
    y = c + a*np.sin(2.0*np.pi*x + b)
    return y

def pulse_model2(p, x):
    a, b, c, d, e = p
    y = c + a*np.sin(2.0*np.pi*x + b) + d*np.sin(4.0*np.pi*x + e)
    return y

def residuals(p, data):
    x, y, xerr, yerr = data
    a, b, c = p
    fprime = 2.0*np.pi*a*np.cos(2*np.pi*x + b)
    weights = yerr*yerr + (fprime*fprime)*xerr*xerr
    resid = (y - pulse_model(p,x))/np.sqrt(weights)
    return resid # Weighted residuals

def residuals2(p, data):
    x, y, xerr, yerr = data
    a, b, c, d, e = p
    fprime = 2.0*np.pi*a*np.cos(2.0*np.pi*x + b) +\
              4.0*np.pi*d*np.cos(4.0*np.pi*x + e)
    weights2 = yerr*yerr + (fprime*fprime)*xerr*xerr
    resid2 = (y - pulse_model2(p,x))/np.sqrt(weights2)
    return resid2 # Weighted residuals

#Read parfile and return relevant spin-parameters
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

def output_toas(toas,dtoas):
    strings = []
    string0 = "FORMAT 1"
    string1 = "HEN 0 "
    strings.append(string0)
    for k in range(len(toas)):
        comm = string1 + str(toas[k]) + " " + str(dtoas[k]) + " @"
        strings.append(comm)
    np.savetxt("param.tim",strings,fmt="%s",delimiter="    ")

def arr_time_corr(t0,f0,fdot0,fddot0,tg,dfg,dfdotg,dfgt,tau,events):

    events = np.array(events)
    modsec = 0
    
    if(len(events)>0):
        
        modsec = 0.5*(fdot0/f0)*(events-t0)**2 +\
            (1./6.)*(fddot0/f0)*(events-t0)**3
                 
        modglitch = 0
        
        if(events[0]>=tg):
            modglitch = (dfg/f0)*(events-tg) +\
            0.5*(dfdotg/f0)*(events-tg)**2 +\
            (dfgt/f0)*(1.0 - np.exp(-(events-tg)/tau))

    events = events + modsec + modglitch

    return events

def fit_mod(ph,pherr,rate,rateerr,redthresh,label,plot):
        
    if(plot=="yes"):
                
        plt.figure()
        plt.errorbar(ph,rate,xerr=pherr,yerr=rateerr,\
                      fmt='ko',label="Profiles: "+label)
        plt.xlabel("Phase [cycles]")
        plt.ylabel(r"Rate [s$^{-1}$]")
        plt.legend(loc="best")

    #Initialise fitting parameters
    ainit = 0.2
    binit = 0.2
    cinit = 1.0

    #Perform pulse profile fitting
    paramsinitial = [ainit, binit, cinit]
    fitobj = kmpfit.Fitter(residuals=residuals,data=(ph,rate,pherr,rateerr))
    fitobj.fit(params0=paramsinitial)

    #Goodness of fit
    chi2min = fitobj.chi2_min
    dof = fitobj.dof
    N = 1000

    #First fit a single sinusoid to determine goodness-of-fit
    if(chi2min/dof<redthresh):

        #Obtain best-fit coefficients of pulse profile
        A = fitobj.params[0]
        B = fitobj.params[1]
        Berr = fitobj.xerror[1]
        C = fitobj.params[2]

        #Best-fit model to pulse profile
        xphase = np.linspace(np.min(ph),np.max(ph),N)
        yfunc = C + A*np.sin(2.0*np.pi*xphase + B)

        #Find peak of pulse profile in the range below numerically
        cutoffmin = 0.2
        cutoffmax = 1.2

        yf = yfunc[xphase<=cutoffmax]
        xp = xphase[xphase<=cutoffmax]
        yf = yf[xp>=cutoffmin]
        xp = xp[xp>=cutoffmin]

        #Pulse profile peak (numerical)
        phasemax = xp[np.argmax(yf)]

        #Error in pulse profile peak
        phasemax_err = Berr/(2.0*np.pi)
        
        if(plot=="yes"):
            plt.plot(xphase,yfunc,'r-',label="Best-fit model")
            plt.show()

    #Fit a double harmonic if previous goodness-of-fit is unacceptable
    if(chi2min/dof>=redthresh):

        ainit = 0.2
        binit = 0.2
        cinit = 1.0
        dinit = 0.1
        einit = 0.2

        paramsinitialnu = [ainit, binit, cinit, dinit, einit]
        fobj = kmpfit.Fitter(residuals=residuals2,data=(ph,rate,pherr,rateerr))
        fobj.fit(params0=paramsinitialnu)

        #Obtain best-fit coefficients of pulse profile
        A = fobj.params[0]
        B = fobj.params[1]
        Berr = fobj.xerror[1]
        C = fobj.params[2]
        D = fobj.params[3]
        E = fobj.params[4]

        #Best-fit model
        xphase = np.linspace(np.min(ph),np.max(ph),N)
        yfunc = C + A*np.sin(2.0*np.pi*xphase + B) +\
                D*np.sin(4.0*np.pi*xphase + E)

        #Find peak of pulse profile in the range below numerically
        cutoffmin = 0.2
        cutoffmax = 1.2
        yf = yfunc[xphase<=cutoffmax]
        xp = xphase[xphase<=cutoffmax]
        yf = yf[xp>=cutoffmin]
        xp = xp[xp>=cutoffmin]

        phasemax = xp[np.argmax(yf)]
        phasemax_err = Berr/(2.0*np.pi)

        chi2min = fobj.chi2_min
        dof = fobj.dof
        
        if(plot=="yes"):
            plt.plot(xphase,yfunc,'r-',label="Best-fit model")
            plt.show()

        if(chi2min/dof>redthresh):

            # print("Fitting procedure not successful with 2 harmonics")
            phasemax = -99
            phasemax_err = -99

    return phasemax, phasemax_err, xphase, yfunc

def phase_fit(t0,f0,fdot0,fddot0,tg,dfg,dfdotg,dfgt,tau,nbin,fname,plot):

    times,dtimes,phimax,phimaxerr,htestsc = [[],[],[],[],[]]
    strate,strateerr = [[],[]]
    for i0 in range(len(fname)):

        hdulist = fits.open(fname[i0])
        ev = hdulist[1].data['TIME']
        hdr = hdulist[1].header
        
        if(fname[i0][0:2]=='ni'):
            
            t0 = 229842845.680
            tstart = hdr['TSTART']
            tstop = hdr['TSTOP']
            mjdref = hdr['MJDREFI'] + hdr['MJDREFF']   
            mjdstart = mjdref + tstart/day
            mjdstop = mjdref + tstop/day
            tobs = 0.5*(mjdstart + mjdstop)
            dtobs = 0.5*(mjdstop - mjdstart)
            obsid = hdr['OBS_ID']

        if(fname[i0][0:2]=='nu'):
            t0 = 325870785.617
            tstart = hdr['TSTART']
            tstop = hdr['TSTOP']
            mjdref = hdr['MJDREFI'] + hdr['MJDREFF']            
            mjdstart = mjdref + tstart/day
            mjdstop = mjdref + tstop/day
            tobs = 0.5*(mjdstart + mjdstop)
            dtobs = 0.5*(mjdstop - mjdstart)
            obsid = hdr['OBS_ID']

        if(fname[i0][0]=='x'):
            t0 = 704562051.801
            tstart = hdr['TSTART']
            tstop = hdr['TSTOP']
            mjdref = hdr['MJDREF']          
            mjdstart = mjdref + tstart/day
            mjdstop = mjdref + tstop/day
            tobs = 0.5*(mjdstart + mjdstop)
            dtobs = 0.5*(mjdstop - mjdstart)
            obsid = hdr['OBS_ID']
    
        if(fname[i0][0:2]=='sw'):
            
            t0 = 609867607.473
            tstart = hdr['TSTART']
            tstop = hdr['TSTOP']
            mjdref = hdr['MJDREFI'] + hdr['MJDREFF']            
            mjdstart = mjdref + tstart/day
            mjdstop = mjdref + tstop/day
            tobs = 0.5*(mjdstart + mjdstop)
            dtobs = 0.5*(mjdstop - mjdstart)
            obsid = hdr['OBS_ID']

        if(len(ev)>0):
            
            ev = arr_time_corr(t0,f0,fdot0,fddot0,tg,dfg,dfdotg,dfgt,tau,ev)
            phase, rate, rerr = fold_events(ev,f0,nbin=nbin,ref_time=t0)
            dp = phase[1]-phase[0]
            pcopy = np.arange(phase[-1]+dp,2.0*phase[-1]+dp,dp)

            pnew = np.append(phase,pcopy)
            xerrnew = np.zeros(len(pnew)) + pnew[1]-pnew[0]
            ratenew = np.append(rate,rate)
            rerrnew = np.append(rerr,rerr)
                        
            lc = Lightcurve(pnew*f0**-1,ratenew,dt=1.0,skip_checks=True)
            csum = np.sum(lc.counts)
            cthresh = 2000.0
            m,htestscore = htest(ratenew,nmax=3)
            hthresh = 10.0
            
            if(htestscore>hthresh and csum>cthresh):
                
                label = obsid + ": " + str(tobs)
                chithresh = 1.2
                pmax,pmaxerr,xph,yf =\
                fit_mod(pnew,xerrnew,ratenew,rerrnew,chithresh,label,plot)
                                
                if(pmax>1):
                    pmax-=1.0
                             
                if(pmax>-99):
                    times.append(tobs)
                    dtimes.append(dtobs)
                    phimax.append(pmax)
                    phimaxerr.append(pmaxerr)
                    htestsc.append(htestscore)
                    
                    normrate = integrate.simps(ratenew,pnew)                    
                    strate.append(ratenew)
                    strateerr.append(rerrnew)
    
    times, phimax, phimaxerr,htestsc,=\
    zip(*sorted(zip(times,phimax,phimaxerr,htestsc)))
    times = np.array(times)
    phimax = np.array(phimax)
    phimaxerr = np.array(phimaxerr)
    htestsc = np.array(htestsc)

    return times,phimax,phimaxerr,htestsc,pnew,strate,strateerr
    
###############################################################################
###############################################################################

#Obtain initial spin parameters from par-file
parfile = 'post_late.par'
t0,f0,fdot0,fddot0 = loadparfile(parfile)
T0 = t0

#Find all event files in directory
fname = []
for file in sorted(glob.glob("*bary_filtered.fits")):
    fname.append(file)
dfg = 0.0
dfdotg = 0.0
tg = 0.0
dfgt = 0.0*dfg
tau = 100.0

# Perform phase-fitting for input event files
plot = "no" #Plot pulse profiles
nbin = 24 #Number of pulse profile bins
times, phimax, phimaxerr, hstat, pnew, strate, strateerr =\
phase_fit(t0,f0,fdot0,fddot0,tg,dfg,dfdotg,dfgt,tau,nbin,fname,plot)

strate = np.array(strate)
strateerr = np.array(strateerr)

# Phase shifts (NICER) post-outburst
mask = np.ones(len(times),dtype=bool)
mask[[2,-5,-4,-3,-2,-1]] = False
times = times[mask]
phimax = phimax[mask]
phimaxerr = phimaxerr[mask]
strate = strate[mask]
strateerr = strateerr[mask]

Z = np.column_stack((times,phimax,phimaxerr))
np.savetxt('phase_resid.dat',Z,fmt="%s",delimiter="    ")

plt.figure()
plt.errorbar(times,phimax,yerr=phimaxerr,fmt='ro',label="Timing residuals")
plt.xlabel("Time [days]",fontsize=14)
plt.ylabel("Pre-fit residual [cycles]",fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(loc="best")
plt.show()

#Stacked pulse profile
strate = np.array(strate)
strateerr = np.array(strateerr)
srate,srateerr = [[],[]]

for k in range(len(strate[0])):
    v = 0
    w = 0
    for j in range(len(strate)):
        v += strate[j][k]
        w += strateerr[j][k]**2
    w = np.sqrt(w)
    srate.append(v)
    srateerr.append(w)

pnew = np.array(pnew)
srate = np.array(srate)
srateerr = np.array(srateerr)

Z = np.column_stack((pnew,srate,srateerr))
np.savetxt("stacked_pprof_post.dat",Z,fmt='%s',delimiter='   ')

plt.figure()
plt.errorbar(pnew,srate,yerr=srateerr,fmt='ko')
plt.show()

