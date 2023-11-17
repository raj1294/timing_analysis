from astropy.io import fits
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from stingray.pulse.search import phaseogram

day = 86400
mjdrefi = 56658.0

#Read parfile and return relevant spin-parameters
def loadparfile(pfile):
    
    with open(parfile) as fi:
        for line in fi:
            line = line.split()
            if(line[0]=='PEPOCH'):
                t0 = np.float64(line[1])
            if(line[0]=='F0'):
                f0 = np.float64(line[1])
            if(line[0]=='F1'):
                fdot0 = np.float64(line[1])
            if(line[0]=='F2'):
                fddot0 = np.float64(line[1])
    
    return t0,f0,fdot0,fddot0

def arrival_time_corr(t0,f0,fdot0,fddot0,events):
    for j in range(len(events)):
        events[j] = events[j] + 0.5*(fdot0/f0)*(events[j]-t0)**2 +\
            (1./6.)*(fddot0/f0)*(events[j]-t0)**3
            
    return events

def phogram(t0,f0,fdot0,fddot0,nbin,ntimes,fname):
    
    ev = []
    for i0 in range(len(fname)):

        hdulist = fits.open(fname[i0])
        events = hdulist[1].data['TIME']
        events = arrival_time_corr(t0,f0,fdot0,fddot0,events)
        for j0 in range(len(events)):
            ev.append(events[j0])
    
    ev = np.array(ev)
    phaseogr, phases, times, additional_info = \
    phaseogram(ev, f0, nph=nbin, nt=ntimes, pepoch=t0)
    
    return times,phases,phaseogr


#Obtain initial spin parameters from par-file
parfile = 'post_outburst.par'
t0,f0,fdot0,fddot0 = loadparfile(parfile)

#Convert to NICER seconds
T0 = t0
t0 = (t0-mjdrefi)*day

# #Copy original event files
# os.system("./cpfiles.sh")

#Find all event files in directory and arrange in order of obs date
fname = []
for file in sorted(glob.glob("ni*.evt")):
    fname.append(file)
f1,f2 = [[],[]]
for k in range(len(fname)):
    if(len(fname[k])==7):
        f1.append(fname[k])
    if(len(fname[k])>7):
        f2.append(fname[k])
fname = f1 + f2

#Parameters for phaseogram and gaussian filtering
window_size = [12,7,5,3,4,6]
sigthresh = [0.1,0.1,0.1,0.6,0.6,0.7]
ntimes = [15,10,8,5,6,2]
nbin = 20

maxph,maxpht,stdpht = [[],[],[]]
mt, pgramunfilt, pgramfilt = [[],[],[]]
lastt = 0
count = 0
for k2 in range(len(window_size)):    
    in1 = np.int64(count)
    in2 = np.int64(count + window_size[k2])
    time,ph,psgram = phogram(t0,f0,fdot0,fddot0,nbin,ntimes[k2],\
                             fname[in1:in2])
    count += window_size[k2]
    psgram = psgram/np.max(psgram)

    mean_ph = 0.5*(ph[1:] + ph[:-1])
    mean_t = 0.5*(time[1:] + time[:-1])
    mean_t = (mean_t - t0)/day
    stdlim = 12
    
    for p in range(len(psgram.T)):
        if(np.sum(psgram.T[p])>0):
            
            argmaxphase = np.argmax(psgram.T[p])
            maxphase = ph[argmaxphase]
            if(argmaxphase<stdlim):
                argmaxphase += len(mean_ph)/2.0
            argmaxphase = np.int64(argmaxphase)
            std = np.std(ph[argmaxphase-stdlim:argmaxphase+stdlim])
            
            if(maxphase<0.5):
                maxphase+=1
            maxph.append(maxphase)
            maxpht.append(mean_t[p])
            stdpht.append(std)
                            
    mt.append(mean_t)
    pgramunfilt.append(psgram)    
    psgram_filt = gaussian_filter(psgram,sigma=sigthresh[k2])
    pgramfilt.append(psgram_filt)
    lastt += mean_t[-1]
    mean_t = mean_t + lastt        

arrpgramunfilt = []
arrpgramfilt = []
arrmt = []

for k3 in range(len(pgramfilt[0])):
    
    inarrunfilt = np.empty(0)
    inarr = np.empty(0)
    for k4 in range(len(pgramfilt)):
        
        outarrunfilt = pgramunfilt[k4][k3]
        outarr = pgramfilt[k4][k3]
        
        intarrunfilt = np.hstack((inarrunfilt,outarrunfilt))
        intarr = np.hstack((inarr,outarr))
        
        inarrunfilt = intarrunfilt
        inarr = intarr        
        
        if(k3==0):
            for k5 in range(len(mt[k4])):
                arrmt.append(mt[k4][k5])  
    
    arrpgramunfilt.append(intarrunfilt)
    arrpgramfilt.append(intarr)
    
arrmt = np.array(arrmt)
mean_ph = np.array(mean_ph)
arrpgramunfilt = np.array(arrpgramunfilt)
arrpgramfilt = np.array(arrpgramfilt)
maxph = np.array(maxph)
maxpht = np.array(maxpht)
stdpht = np.array(stdpht)

maxph[23:] = maxph[23:] + 1.0
maxph[-5:] = maxph[-5:] - 1.0
# maxph = np.delete(maxph,-2)
# maxpht = np.delete(maxpht,-2)
# stdpht = np.delete(stdpht,-2)
# maxph = np.delete(maxph,-1)
# maxpht = np.delete(maxpht,-1)
# stdpht = np.delete(stdpht,-1)

mean_ph2 = mean_ph + 2.0
maxphmin = maxph - stdpht*0.5
maxphmax = maxph + stdpht*0.5
arrmt = arrmt + T0
maxpht = maxpht + T0

plt.figure()
plt.title("Phaseogram smoothed with a Gaussian kernel: width of 1.5$\sigma$")
plt.pcolormesh(mean_ph,arrmt,arrpgramfilt.T,cmap='viridis',\
                vmin=np.min(arrpgramfilt.T),vmax=np.max(arrpgramfilt.T))
plt.pcolormesh(mean_ph2,arrmt,arrpgramfilt.T,cmap='viridis',\
                vmin=np.min(arrpgramfilt.T),vmax=np.max(arrpgramfilt.T))
plt.plot(maxph,maxpht,'rD',linewidth=2,label="Peak of pulse phase")
plt.plot(maxphmin,maxpht,'k-',linewidth=2,label="Standard deviation")
plt.plot(maxphmax,maxpht,'k-',linewidth=2)
plt.xlabel("Phase [cycles]",fontsize=14)
plt.ylabel("Time [MJD]",fontsize=14)
plt.ylim(59063,59125)
clb = plt.colorbar()
clb.set_label('Normalised counts',fontsize=14)
plt.legend(loc="best")
plt.show()




