import warnings
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from stingray import AveragedPowerspectrum, EventList
from stingray import AveragedCrossspectrum, Lightcurve
from stingray.varenergyspectrum import LagSpectrum
import glob
from astropy.io import fits
from kapteyn import kmpfit
from scipy import integrate, signal    
    
warnings.filterwarnings("ignore")

def remove_nans(arrnans):
        
    newarrnanslist = []
    for qdnans in range(len(arrnans)):
                                                                              
        isnanarr = np.isnan(arrnans[qdnans])
        for qdnans2 in range(len(isnanarr)):   
            
            if(isnanarr[qdnans2]==True):
                
                newindex = np.arange(0,len(arrnans),1)
                
                for qdnans3 in range(len(newindex)):
                    arrnans[qdnans3][qdnans2] = -1e10
        
        newarrnanslist.append(arrnans[qdnans])

    for pindnan in range(len(newarrnanslist)):    
        newarrnanslist[pindnan] = np.array(newarrnanslist[pindnan])
        newarrnanslist[pindnan] = newarrnanslist[pindnan]\
                                  [newarrnanslist[pindnan]>-1e9]
    newarrnanslist = np.array(newarrnanslist)
    return newarrnanslist
    
def ignore_btis(arrays,tS,tE):
        
    tref = arrays[-1]   
    oldarrays_list,newarrays_list = [[],[]]
        
    for qd in range(len(arrays)):
                                                                                
        #Identify BTIs
        for qd2 in range(len(tS)-1):
            
            btiS = tE[qd2]
            btiE = tS[qd2+1]
                        
            for qd3 in range(len(tref)):
                
                if(tref[qd3]>=btiS and tref[qd3]<=btiE\
                   and qd!=len(arrays)-1):
                    
                    newindarray = np.arange(0,len(arrays),1)
                    for qd4 in range(len(newindarray)):
                        arrays[qd4][qd3] = -1e10
                    
        #Remove BTIs
        if(qd!=len(arrays)-1):
            newarray = arrays[qd][arrays[qd]>-100]
            newarrays_list.append(newarray)
        oldarrays_list.append(arrays[qd])
        
    for pind in range(len(newarrays_list)):    
        newarrays_list[pind] = np.array(newarrays_list[pind])
    newarrays_list = np.array(newarrays_list)
    for pind2 in range(len(oldarrays_list)):
        oldarrays_list[pind2] = np.array(oldarrays_list[pind2])
    oldarrays_list = np.array(oldarrays_list)
    
    return oldarrays_list, newarrays_list

def rect_window(ewref):
    
    ywin = []    
    for qw in range(len(ewref)):
        
        if(ewref[qw]==0):
            ywin.append(0)
        if(ewref[qw]>0):
            ywin.append(1)
    ywin = np.array(ywin)
        
    return ywin

#Function to average PSD and CPSD over M segments
def Pbin(Msegs,freqsarr,Pxarr,Pyarr,Cxyarr):
    
    favg,Pxavg,dPxavg,Pyavg,dPyavg,Cxyavg,dCxyavg = [[],[],[],[],[],[],[]]
                    
    #Average CPSD and PSD over M segments
    for iav in range(np.shape(Cxyarr)[1]):
        
        px = 0
        py = 0
        cxy = 0
        freq = freqsarr[0][iav]
        
        for jav in range(np.shape(Cxyarr)[0]):
            
            cxy += Cxyarr[jav][iav]/Msegs
            px += Pxarr[jav][iav]/Msegs
            py += Pyarr[jav][iav]/Msegs
                
        favg.append(freq)
        Cxyavg.append(cxy)
        Pxavg.append(px)
        Pyavg.append(py)
        dPxavg.append(px/np.sqrt(Msegs))
        dPyavg.append(py/np.sqrt(Msegs))
        dCxyavg.append(cxy/np.sqrt(Msegs))
    
    favg = np.array(favg)
    Pxavg = np.array(np.real(Pxavg))
    Pyavg = np.array(np.real(Pyavg))
    dPxavg = np.array(np.real(dPxavg))
    dPyavg = np.array(np.real(dPyavg))
    Cxyavg = np.array(Cxyavg)
    dCxyavg = np.array(dCxyavg)
    
    #Return averaged quantities
    return favg,Pxavg,dPxavg,Pyavg,dPyavg,Cxyavg,dCxyavg

#Function to bin PSD and CPSD in frequency space
def fbin(bfact,farr,PXarr,PYarr,CXYarr,dPXarr,dPYarr,dCXYarr):
    
    avgf,avgpx,davgpx,avgpy,davgpy,avgcxy,davgcxy,Karr =\
    [[],[],[],[],[],[],[],[]]
        
    bmin = 0
    bmax = 0
    while(bmax<(len(farr))):
        
        if((bmax-bmin)!=0):
            bmin = bmax
        fmax = bfact*farr[bmax]        
        for index in range(bmin,len(farr)):
            if(farr[index]>=fmax):
                bmax = index
                break
            if(index==len(farr)-1 and farr[index]<=fmax):
                bmax = index + 1
                break
        if((bmax-bmin)==0):
            bfact+=0.1
                
        #Append relevant quantities        
        af = 0
        apx = 0
        apy = 0
        acxy = 0
        errpx = 0
        errpy = 0
        errcxy = 0
                
        for k2bin in range(bmin,bmax):
            af += farr[k2bin]
            apx += PXarr[k2bin]
            apy += PYarr[k2bin]
            acxy += CXYarr[k2bin]
            errpx += dPXarr[k2bin]**2
            errpy += dPYarr[k2bin]**2
            errcxy += dCXYarr[k2bin]**2
        
        af/=(bmax-bmin)
        apx/=(bmax-bmin)
        apy/=(bmax-bmin)
        acxy/=(bmax-bmin)
        errpx = np.sqrt(errpx)/(bmax-bmin)
        errpy = np.sqrt(errpy)/(bmax-bmin)
        errcxy = np.sqrt(errcxy)/(bmax-bmin)
        
        avgf.append(af)
        avgpx.append(apx)
        avgpy.append(apy)
        avgcxy.append(acxy)
        davgpx.append(errpx)
        davgpy.append(errpy)
        davgcxy.append(errcxy)        
        Karr.append(bmax-bmin)
            
    #Return binned quantities
    return avgf,avgpx,avgpy,avgcxy,davgpx,davgpy,davgcxy,Karr

#Estimate time lag in Fourier domain (Uttley et al. 2014)
def time_lag_func(lc,lcerr,reflc,reflcerr,lcbkg,refbkg,ywindow,\
                  Mseg,bfactor,dt,stat):
    
    freqs,Px,Py,Cxy = [[],[],[],[]]
    
    # Compute noise level depending on whether the counting statistics
    # are Poissonian or not 
    fnyq = 0.5*(dt**-1)
    Pnoise = 0
    Prefnoise = 0
    Msegnew = 0

    # Average power spectrum and cross spectrum over M segments
    for k3t in range(Mseg):
        
        # Split LC into M equal segments
        div = int(len(reflc)/Mseg)
        lctemp = lc[k3t*div:(k3t+1)*div]
        lctemperr = lcerr[k3t*div:(k3t+1)*div]
        reflctemp = reflc[k3t*div:(k3t+1)*div]
        reflctemperr = reflcerr[k3t*div:(k3t+1)*div]
        lcbkgtemp = lcbkg[k3t*div:(k3t+1)*div]
        refbkgtemp = refbkg[k3t*div:(k3t+1)*div]
        ywindowtemp = ywindow[k3t*div:(k3t+1)*div]
        
        if(stat=="Poissonian"):
            
            Pnoise += (2*(np.mean(lctemp) +\
                       np.mean(lcbkgtemp))/(np.mean(lctemp))**2)
            Prefnoise += (2*(np.mean(reflctemp) +\
                          np.mean(refbkgtemp))/(np.mean(reflctemp))**2)
        
        if(stat!="Poissonian"):
            
            errsq = np.sum(lctemperr**2)/len(lctemperr)
            errrefsq = np.sum(reflctemperr**2)/len(reflctemperr)
            
            Pnoise += errsq/(fnyq*(np.mean(lctemp))**2)
            Prefnoise += errrefsq/(fnyq*(np.mean(reflctemp))**2) 
        
        if(np.sum(lctemp)>0):
            
            # FFT of comparison-band LC
            Xn = fft.fft(lctemp) 
            fxn = fft.fftfreq(len(lctemp),d=dt)
            
            # FFT of reference-band LC
            Yn = fft.fft(reflctemp) 
            fyn = fft.fftfreq(len(reflctemp),d=dt)
            
            # FFT of window function
            Wn = fft.fft(ywindowtemp)
            fwn = fft.fftfreq(len(ywindowtemp),d=dt)
            
            remx,Xn = signal.deconvolve(Xn,Wn)
            remy,Yn = signal.deconvolve(Yn,Wn)
                        
            Xn = Xn[fxn>0]
            Yn = Yn[fyn>0]
            Wn = Wn[fwn>0]
            fxn = fxn[fxn>0]
            fyn = fyn[fyn>0]
            fwn = fwn[fwn>0]
                                    
            # Compute PSD and CPSD with 
            # rms-squared normalisation for each segment
            normpsdx = (2.0*dt)/((len(lctemp))*(np.mean(lctemp))**2)
            normpsdy = (2.0*dt)/((len(reflctemp))*(np.mean(reflctemp))**2)
            normcross = (2.0*dt)/((len(lctemp))*(np.mean(lctemp))*\
                                 (np.mean(reflctemp)))
                    
            Psdx = normpsdx*((np.conj(Xn))*(Xn))
            Psdy = normpsdy*((np.conj(Yn))*(Yn))
            Crossxy = normcross*((np.conj(Yn))*(Xn))
                    
            # Append CPSD and PSD for each segment to 
            # pass to functions for averaging and binning
            if(len(Crossxy)>0 and len(Psdx)>0 and len(Psdy)>0):

                freqs.append(fxn)
                Px.append(Psdx)
                Py.append(Psdy)
                Cxy.append(Crossxy)
                Msegnew += 1
        
    #Average Pnoise and Prefnoise
    Pnoise /= Msegnew
    Prefnoise /= Msegnew
    
    if(len(Cxy)>0):
        
        Mseg = Msegnew
        freqs = np.array(freqs)
        Px = np.array(Px)
        Py = np.array(Py)
        Cxy = np.array(Cxy)
            
        # Average PSDs and CPSDs over M segements
        favg,Pxavg,dPxavg,Pyavg,dPyavg,Cxyavg,dCxyavg =\
        Pbin(Mseg,freqs,Px,Py,Cxy)
        
        fb = favg
        avgPx = Pxavg
        avgPy = Pyavg
        avgCxy = Cxyavg
        avgPxerr = dPxavg
        avgPyerr = dPyavg
        avgCxyerr = dCxyavg
        Karr = np.ones(len(Pxavg))

        # Implement frequency dependent binning of averaged PSDs and CPSDs
        if(bfactor>1):
            fb,avgPx,avgPy,avgCxy,avgPxerr,avgPyerr,avgCxyerr,Karr =\
            fbin(bfactor,favg,Pxavg,Pyavg,Cxyavg,dPxavg,dPyavg,dCxyavg)
        
        avgPx = np.real(avgPx)
        avgPy = np.real(avgPy)
        avgPx = np.array(avgPx)
        avgPxerr = np.array(avgPxerr)
        avgPy = np.array(avgPy)
        avgPyerr = np.array(avgPyerr)
        avgCxy = np.array(avgCxy)
        avgCxyerr = np.array(avgCxyerr)
        Karr = np.array(Karr)
        fb = np.array(fb)
                                
        # Frequency bin widths
        if(len(fb)>1):
            dfb = 0.5*(fb[1:]-fb[0:-1])
            
        if(len(fb)<=1):
            dfb = np.zeros(1)
                
        # Averaged number of samples
        nsamples = Mseg*Karr
            
        # Noise level of CPSD amplitude
        nbias = ((avgPx-Pnoise)*Prefnoise + (avgPy-Prefnoise)*Pnoise +\
                (Pnoise*Prefnoise))/nsamples
        
        # Complex-valued CPSD amplitude
        Cxyamp = (np.real(avgCxy))**2 + (np.imag(avgCxy))**2 - nbias
        
        # Coherence
        coherence = Cxyamp/((avgPx)*(avgPy)) #Raw
        intcoherence = Cxyamp/((avgPx-Pnoise)*(avgPy-Prefnoise)) #Intrinsic
        
        # Statistical uncertainty on raw coherence
        dcoherence = ((2.0/(nsamples))**(0.5))*(1 - intcoherence**2)/\
                     (abs(intcoherence))
                
        # Uncertainty in intrinsic coherence (from Vaughan and Nowak 1997)
        intcoherr = np.zeros(len(intcoherence))
        arbfact = 3 
        for u in range(len(coherence)):
                        
            # High powers, high measured coherence         
            cond1 = (arbfact*Pnoise)/(np.sqrt(nsamples[u]))
            cond2 = (arbfact*Prefnoise)/(np.sqrt(nsamples[u]))
            cond3 = (arbfact*nbias[u])/((avgPx[u])*(avgPy[u]))
            
            if((avgPx[u]-Pnoise)>cond1 and (avgPy[u]-Prefnoise)>cond2\
                and coherence[u]>cond3):
                                
                intcoherr[u] = ((nsamples[u])**-0.5)*\
                               (np.sqrt((2*nsamples[u]*nbias[u]**2)/\
                               (Cxyamp[u] - nbias[u])**2 +\
                               ((Pnoise)/(avgPx[u]-Pnoise))**2 +\
                               ((Prefnoise)/(avgPy[u]-Prefnoise))**2 +\
                               (nsamples[u]*dcoherence[u]**2)/\
                               (intcoherence[u]**2)))
                                    
                intcoherr[u] *= intcoherence[u]
                intcoherr[u] = abs(intcoherr[u])
                if(np.isnan(intcoherr[u])=='True'):
                    intcoherr[u] = 0
            
            # High powers, low measured coherence 
            else:    
                intcoherr[u] = np.sqrt(Prefnoise**2/(avgPx[u]-Prefnoise)**2/\
                nsamples[u] + Pnoise**2/(avgPy[u]-Pnoise)**2/\
                nsamples[u] + (dcoherence[u]/intcoherence[u])**2)
                intcoherr[u] *= intcoherence[u]
                intcoherr[u] = abs(intcoherr[u])
                if(np.isnan(intcoherr[u])=='True'):
                    intcoherr[u] = 0
        
        # Compute phase lag as a function of frequency between the 
        # two energy bands        
        Cxyimag = np.imag(avgCxy)
        Cxyreal = np.real(avgCxy)
        phaselag = np.zeros(len(Cxyimag))
        dphaselag = np.zeros(len(Cxyimag))
                
        for hp3 in range(len(phaselag)):
                        
            #Clockwise
            phaselag[hp3] = np.arctan(Cxyimag[hp3]/-Cxyreal[hp3])
            div = phaselag[hp3]/np.pi
            
            #Ensure phase lag is confined to between -pi to pi
            if(div>1):
                            
                divs = str(div)
                divnum = int(divs.split(".")[0])
                                
                #Should be between 0 to 1
                if((divnum-1)%3==0):
                    div -= divnum
                
                #Should be between 0 to 1
                if(divnum%3==0):
                    div -= divnum

                #Should be between -1 to 0
                if((divnum+1)%3==0):
                    div += (divnum+1)
                
            if(div<-1):
                
                divs = str(div)
                divnum = int(divs.split(".")[0])
                
                #Should be between 0 to 1
                if((divnum-1)%3==0):
                    div -= (divnum-1)
                
                #Should be between 0 to 1
                if(divnum%3==0):
                    div -= divnum

                #Should be between -1 to 0
                if((divnum+1)%3==0):
                    div -= divnum
            
            phaselag[hp3] = div*np.pi
            
            coherence[hp3] = intcoherence[hp3]**2
            # if(coherence[hp3]>=1):
            #     coherence[hp3] = 1
            
            # dphaselag[hp3] = np.sqrt((1.0-intcoh)/\
            #                  (2.0*nsamples[hp3]*intcoh))

            dphaselag[hp3] = np.sqrt((1.0-coherence[hp3])/\
                             (2.0*nsamples[hp3]*coherence[hp3]))

    return fb,dfb,phaselag,dphaselag,intcoherence,intcoherr

#Function to model Averaged PSD
def plmod(pars, xdata):
    
    amplitude,alpha = pars
    ymod = amplitude*((xdata)**-alpha)
    
    if(alpha<0 or amplitude<-0.5):
        ymod = np.zeros(len(xdata))
        
    return ymod

#Residuals
def residuals(pars, data):
    
    xdata, ydata, ydataerr = data
    amplitude,alpha = pars
    resid = (ydata - plmod(pars,xdata))/ydataerr
    
    return resid 

#Replace gaps with Poisson noise 
def gapfill(gtist,gtiend,timlc,ratelc,errorlc):
    
    wsize = 200
    binsize = timlc[1]-timlc[0]
    
    for gtik in range(len(gtist)):
        gapsize = int((gtiend[gtik]-gtist[gtik])/binsize)
        for tm in range(len(timlc)):
            if(timlc[tm]>=gtist[gtik] and timlc[tm]<=gtiend[gtik]):
                
                #Arrays for estimating mean count-rate
                arr1 = ratelc[tm-wsize:tm]
                arr2 = ratelc[tm+gapsize:tm+gapsize+wsize]
                
                isnan1 = np.isnan(arr1)
                isnan2 = np.isnan(arr2)
                arr1 = arr1[isnan1==False]
                arr2 = arr2[isnan2==False]
    
                meanrate = np.median(np.hstack((arr1,arr2)))
                if(gtik==0):
                    meanrate = np.median(arr2)
                if(gtik==len(gtist)-1):
                    meanrate = np.median(arr1)
                ct = int(meanrate*binsize)
                ratelc[tm] = np.random.poisson(ct)/binsize
                errorlc[tm] = np.sqrt(ct)/binsize
    
    return timlc,ratelc,errorlc
    
#Timmer & Koenig method to generate fake LC
def drawsamp(frequency,amp,expnt):
    
    psdomega = amp*((frequency)**(-expnt))
        
    r1 = np.random.normal(0.0,scale=np.sqrt(abs(0.5*psdomega)))
    r2 = np.random.normal(0.0,scale=np.sqrt(abs(0.5*psdomega)))
    
    compnumpos = complex(r1,r2)
    compnumneg = np.conj(compnumpos)
    
    return compnumpos,compnumneg

#Estimation of random noise contribution to lag energy spectrum
def mcmc(bin_time_mcmc,tdur,tsegsize,geom_rebin,freqmin,freqmax,\
         A1,ind1,A2,ind2,murate1,murate2,plts):
        
    #Draw randomly from best-fit PSD and inverse FFT to generate LC
    omegamin = 1./tdur
    omegamax = 0.5*(bin_time_mcmc**-1)
    domega = omegamin
    omega = np.arange(omegamin,omegamax+domega,domega)  
    
    complexfft1pos = np.zeros(int(len(omega))).astype(complex)    
    complexfft1neg = np.zeros(int(len(omega))).astype(complex)    
    complexfft2pos = np.zeros(int(len(omega))).astype(complex)
    complexfft2neg = np.zeros(int(len(omega))).astype(complex)

    for irand in range(int(len(omega))):
        
        compnumber_pos1,compnumber_neg1 = drawsamp(omega[irand],A1,ind1)
        compnumber_pos2,compnumber_neg2 = drawsamp(omega[irand],A2,ind2)
        
        complexfft1pos[irand] = compnumber_pos1
        complexfft1neg[irand] = compnumber_neg1
        complexfft2pos[irand] = compnumber_pos2
        complexfft2neg[irand] = compnumber_neg2

        if(irand==int(len(omega))-1):
            complexfft1neg[irand] = np.real(complexfft1neg[irand])
            complexfft2neg[irand] = np.real(complexfft2neg[irand])
            
    complexfft1neg = np.flip(complexfft1neg)
    complexfft1pos = np.insert(complexfft1pos,0,complex(2*murate1))
    complexfft1 = np.hstack((complexfft1pos,complexfft1neg))
    complexfft2neg = np.flip(complexfft2neg)
    complexfft2pos = np.insert(complexfft2pos,0,complex(2*murate2))
    complexfft2 = np.hstack((complexfft2pos,complexfft2neg))
        
    #Artificial LCs generated from PSDs (Timmer & Köenig 1995)
    counts1 = np.fft.ifft(complexfft1)
    counts1 = np.real(counts1)    
    counts1 = counts1[1:]
    counts2 = np.fft.ifft(complexfft2)
    counts2 = np.real(counts2)
    counts2 = counts2[1:]    
    error1 = np.sqrt(np.random.poisson\
             (np.int64(abs(counts1)*bin_time_mcmc)))/\
             bin_time_mcmc
    error2 = np.sqrt(np.random.poisson\
             (np.int64(abs(counts2)*bin_time_mcmc)))/\
             bin_time_mcmc
    times = np.arange(0,bin_time_mcmc*len(counts1),bin_time_mcmc)
    
    #Ensure positive values for counts while retaining the shape
    ct1_min = abs(np.min(counts1))
    ct1_max = abs(np.min(counts2))    
    counts1 = (counts1 + ct1_min)*bin_time_mcmc
    counts2 = (counts2 + ct1_max)*bin_time_mcmc
    error1 = error1*bin_time_mcmc
    error2 = error2*bin_time_mcmc
        
    if(plts=="yes"):
        
        plt.errorbar(times,counts1,yerr=error1,fmt='k-')
        plt.errorbar(times,counts2,yerr=error2,fmt='r-')
        plt.show()
        
    #Lags from stingray
    lcref_sim = Lightcurve(times,counts=counts1,err=error1,\
                           dt=bin_time_mcmc)
    lccomp_sim = Lightcurve(times,counts=counts2,err=error2,\
                            dt=bin_time_mcmc)
    evref_sim = EventList.from_lc(lcref_sim)
    evcomp_sim = EventList.from_lc(lccomp_sim)
            
    CSAsim = AveragedCrossspectrum.from_events(evref_sim,evcomp_sim,\
             segment_size=tsegsize,norm="frac",use_common_mean=True,\
             dt=bin_time_mcmc)
    CSAsim = CSAsim.rebin_log(geom_rebin)
    
    freq_fake = CSAsim.freq
    lag_fake, lag_e_fake = CSAsim.time_lag()
    
    lag_fake = lag_fake[freq_fake>=freqmin]
    freq_fake = freq_fake[freq_fake>=freqmin]
    lag_fake = lag_fake[freq_fake<=freqmax]
    freq_fake = freq_fake[freq_fake<=freqmax]
    
    lg_fake = np.mean(lag_fake)
    
    return freq_fake, lg_fake

def lag_spec(evfile,fbmin,fbmax,segment_size,btimespec,egrid,refemin,refemax):
    
    events = EventList.read(evfile, "hea",\
                            additional_columns=["DET_ID"])

    lgspec = LagSpectrum(events,\
                         freq_interval=[fbmin,fbmax],\
                         segment_size=segment_size,\
                         bin_time=btimespec,\
                         energy_spec=egrid,\
                         ref_band=[refemin,refemax])
        
    energiesspec = lgspec.energy
    energiesspec_err = np.diff(lgspec.energy_intervals, axis=1).flatten() / 2.
    lgespec = lgspec.spectrum
    lgespecerr = lgspec.spectrum_error
    
    return energiesspec,energiesspec_err,lgespec,lgespecerr

def filteredlc(softbandlc,hardbandlc,freqmin,freqmax,binsize,telapse):
        
    Xnfilt = fft.fft(softbandlc) 
    fxnfilt = fft.fftfreq(len(softbandlc),d=binsize)
    
    Xnpos = Xnfilt[fxnfilt>=freqmin]
    fxnpos = fxnfilt[fxnfilt>=freqmin]
    Xnpos = Xnpos[fxnpos<=freqmax]
    fxnpos = fxnpos[fxnpos<=freqmax]
    Xnneg = Xnfilt[fxnfilt>=-freqmax]
    fxnneg = fxnfilt[fxnfilt>=-freqmax]
    Xnneg = Xnneg[fxnneg<=-freqmin]
    fxnneg = fxnneg[fxnneg<=-freqmin]
    
    Xnneg = np.flip(Xnneg)
    compxn = np.hstack((Xnpos,Xnneg))
    compxn = np.insert(compxn,0,complex(Xnfilt[0]))
    freqxn = np.hstack((fxnpos,fxnneg))
    freqxn = np.insert(freqxn,0,fxnfilt[0])
    
    countsx = np.fft.ifft(compxn)
    countsx = np.real(countsx)    
    
    Ynfilt = fft.fft(hardbandlc) 
    fynfilt = fft.fftfreq(len(hardbandlc),d=binsize)
        
    Ynpos = Ynfilt[fynfilt>=freqmin]
    fynpos = fynfilt[fynfilt>=freqmin]
    Ynpos = Ynpos[fynpos<=freqmax]
    fynpos = fynpos[fynpos<=freqmax]
    Ynneg = Ynfilt[fynfilt>=-freqmax]
    fynneg = fynfilt[fynfilt>=-freqmax]
    Ynneg = Ynneg[fynneg<=-freqmin]
    fynneg = fynneg[fynneg<=-freqmin]
    
    Ynneg = np.flip(Ynneg)
    compyn = np.hstack((Ynpos,Ynneg))
    compyn = np.insert(compyn,0,complex(Ynfilt[0]))  
    freqyn = np.hstack((fynpos,fynneg))
    freqyn = np.insert(freqyn,0,fynfilt[0])
    
    countsy = np.fft.ifft(compyn)
    countsy = np.real(countsy)
    timefilt = np.linspace(0,telapse,len(countsy))
    
    return timefilt, countsx, countsy
    
ks = 1000 #ks in seconds
reblog = 0.0 # Geometric binning factor
geombin = 1.0 #Geometric binning factor
Nsegments = 8 #Number of LC segments to average PSD
Ntrialmcmc = 1 #Number of MCMC simulations
Nenergies = 12
stats = "NP" #Poisson statistics for LC
Plot = "no" #Plot
Plotmcmc = "no" #Plot MCMC generated LCs

obsid = "0405690201"
# obsid = "0405690501"
# obsid = "0921360101"
# obsid = "0921360201"

#Frequency bins
# fbinmin = [1e-5,1e-4,4e-4]
# fbinmax = [1e-4,4e-4,1e-3]
fbinmin = [1e-4]
fbinmax = [5e-4]
fbinmin = np.array(fbinmin)
fbinmax = np.array(fbinmax)

plt.figure(figsize=(5,5))
for k in range(len(fbinmin)):
    
    enlag,denlag,mlag,mlagerr,mlagS,mlagerrS,mufakelag,fakelagerr,\
    = [[],[],[],[],[],[],[],[]]
    
    for k2 in range(Nenergies-1):
        
        evcomp,rcomp,errcomp,rcompbkg,errcompbkg =\
        [[],[],[],[],[]]
        evref,rref,errref,rrefbkg,errrefbkg,ywref =\
        [[],[],[],[],[],[]]
        
        nobskey = "epn*net*obs*" + str(obsid) + "_1_*en4*ref*.lc"
        ennum = k2+1
        
        for reflcfile in sorted(glob.glob(nobskey)):
                                    
            ObsId = reflcfile.split("obs")[1].split("_")[0]
            
            reflcfile = "epn_net_obs" + ObsId + "_1_en" +\
                         str(ennum) + "_ref.lc"
                        
            refbkgfile = "epn_bkg_obs" + ObsId + "_1_en" +\
                         str(ennum) + "_ref.lc"
            
            refevfile = "epn_net_obs" + ObsId + "_1_en" +\
                         str(ennum) + "_ref.fits"
    
            complcfile = "epn_net_obs" + ObsId + "_1_en" +\
                         str(ennum) + "_comp.lc"
                         
            compbkgfile = "epn_bkg_obs" + ObsId + "_1_en" +\
                          str(ennum) + "_comp.lc"
            
            compevfile = "epn_net_obs" + ObsId + "_1_en" +\
                         str(ennum) + "_comp.fits"
                        
            #Reference band (source LC)
            hdulistref = fits.open(reflcfile)
            dataref = hdulistref[1].data
            timeref = dataref['TIME']  
            rateref = dataref['RATE']
            errorref = dataref['ERROR']
            bsize = hdulistref[1].header['TIMEDEL']
            inst = hdulistref[1].header['INSTRUME']
                                            
            #Reference band (events)
            hdulistrefev = fits.open(refevfile)
            datarefev = hdulistrefev[1].data
            evrefR = datarefev['TIME']
            evrefR = EventList(time=evrefR)
                    
            #Reference band (background LC)
            hdulistref_bkg = fits.open(refbkgfile)
            dataref_bkg = hdulistref_bkg[1].data
            raterefbkg = dataref_bkg['RATE']
            errorrefbkg = dataref_bkg['ERROR']
            
            #Comparison band (source LC)
            hdulistcomp = fits.open(complcfile)
            datacomp = hdulistcomp[1].data
            headercomp = hdulistcomp[1].header
            ratecomp = datacomp['RATE']
            errorcomp = datacomp['ERROR']
                        
            #Comparison band (events)
            hdulistcompev = fits.open(compevfile)
            datacompev = hdulistcompev[1].data
            evcompR = datacompev['TIME']
            evcompR = EventList(time=evcompR)
            
            #Comparison band (background LC)
            hdulistbkg = fits.open(compbkgfile)
            databkg = hdulistbkg[1].data
            ratecompbkg = databkg['RATE']
            errorcompbkg = databkg['ERROR']
            dateref = hdulistbkg[0].header['DATE-OBS']
            
            #GTIs (reference-band)
            tstartR = hdulistref[2].data['START']
            tstopR = hdulistref[2].data['STOP']
            
            #GTIs (comparison-band)
            tstartC = hdulistcomp[2].data['START']
            tstopC = hdulistcomp[2].data['STOP']
            
            #Energy bins
            energymin = float(hdulistcomp[1].header['CHANMIN'])/1000.
            energymax = float(hdulistcomp[1].header['CHANMAX'])/1000.
            energyminref = float(hdulistref[1].header['CHANMIN'])/1000.
            energymaxref = float(hdulistref[1].header['CHANMAX'])/1000.
                
            energies = 0.5*(energymin+energymax)
            denergies = 0.5*(energymax-energymin)
            energiesref = 0.5*(energyminref+energymaxref)
            denergiesref = 0.5*(energymaxref-energyminref)
                    
            #ObsID and Date
            obsid = hdulistref_bkg[0].header['OBS_ID']
            date = hdulistref_bkg[0].header['DATE-OBS']
            
            if(inst=='EPN'):
                            
                #Remove BTIs and NANs from reference band and comparison band
                arraysR = np.transpose(np.column_stack((rateref,errorref,\
                          raterefbkg,errorrefbkg,ratecomp,errorcomp,\
                          ratecompbkg,errorcompbkg,timeref)))
                arraysR, arraysN = ignore_btis(arraysR,tstartR,tstopR)
                
                rateref,errorref,raterefbkg,\
                errorrefbkg,ratecomp,errorcomp,ratecompbkg,errorcompbkg =\
                arraysN

                # #Add a window
                # arraysW = np.transpose(np.column_stack((arraysR[4],\
                #                                         arraysR[5])))
                # ywindowR = rect_window(arraysW,tstartR,tstopR)   
                
                #Concatenate (optional)
                for k4 in range(len(rateref)):
                                        
                    rref.append(rateref[k4])
                    errref.append(errorref[k4])
                    rrefbkg.append(raterefbkg[k4])
                    errrefbkg.append(errorrefbkg[k4])
                    
                    rcomp.append(ratecomp[k4])
                    errcomp.append(errorcomp[k4])
                    rcompbkg.append(ratecompbkg[k4])
                    errcompbkg.append(errorcompbkg[k4])
                    # ywref.append(ywindowR[k4])
                
        ratecomp = np.array(rcomp)
        errorcomp = np.array(errcomp)
        ratecompbkg = np.array(rcompbkg)
        errorcompbkg = np.array(errcompbkg)
        rateref = np.array(rref)
        errorref = np.array(errref)
        raterefbkg = np.array(rrefbkg)
        errorrefbkg = np.array(errrefbkg)
        ywindowR = np.array(ywref)
        timeref = np.arange(0,len(ratecomp),1)*bsize
        timecomp = timeref
        telapse = timecomp[-1]-timecomp[0]
        
        ywindowR = rect_window(errorref)
                                            
        #Mean and RMS of rate
        muref = np.mean(rateref)
        dmuref = np.sum(errorref**2)/len(rateref)
        mucomp = np.mean(ratecomp)
        dmucomp = np.sum(errorcomp**2)/len(ratecomp)
                        
        laben = str(float(energymin)) + "-" + str(float(energymax))
        labenref = str(float(energyminref)) + "-" + str(float(energymaxref))

        labelsrcref = "Reference band LC: " + labenref +\
                      " keV: " + dateref
        labelsrccomp = "Comparison band LC: " + laben +\
                        " keV: " + dateref
        labebkgref = "Reference band: Background only: " + labenref +\
                     " keV"
        labebkgcomp = "Comparison band: Background only: " + laben +\
                      " keV"
        
        # tfilt,ratecomp_filt,rateref_filt =\
        # filteredlc(ratecomp,rateref,fbinmin[k],fbinmax[k],bsize,\
        #            telapse)
        
        # #Plot LCs
        # if (k2==0):
        #     plt.errorbar(timeref/ks,rateref,yerr=errorref,fmt='m--',\
        #                  label=labelsrcref)
        # plt.errorbar(timecomp/ks,ratecomp,yerr=errorcomp)
        # # plt.errorbar(timecomp/ks,ratecompbkg,\
        # #              yerr=errorcompbkg,fmt='g--',\
        # #              label=labebkgcomp)
        # # plt.plot(timecomp/ks,ywindowR,'g-')
        # plt.tick_params(axis='both', which='major', labelsize=18)
        # plt.legend(loc="best")
        # plt.title("XMM-Newton (EPIC-PN) lightcurves: ",\
        #           fontsize=18)
        # plt.ylabel("Count rate [s$^{-1}$]",fontsize=18)
        # plt.xlabel("Time [ks]",fontsize=18)
        # plt.show()
        
        countcomp = ratecomp*bsize
        countcomp_err = errorcomp*bsize
        countref = rateref*bsize
        countref_err = errorref*bsize
        countcompbkg = ratecompbkg*bsize
        countrefbkg = raterefbkg*bsize
                            
        lccomp = Lightcurve(timecomp,countcomp,error=countcomp_err,\
                            dt=bsize)
        lcref = Lightcurve(timeref,countref,error=countref_err,\
                           dt=bsize)
            
        segsizest = (lccomp.time[-1]-lccomp.time[0]+bsize)/Nsegments
        evcomplc = EventList.from_lc(lccomp)   
        evreflc = EventList.from_lc(lcref)
                                        
        #Compute PSDs
        psdcomp = AveragedPowerspectrum.from_lightcurve(lccomp,\
                  segment_size=segsizest,norm="frac")
        psdref = AveragedPowerspectrum.from_lightcurve(lcref,\
                 segment_size=segsizest,norm="frac")
        psdref = psdref.rebin_log(reblog)
        psdcomp = psdcomp.rebin_log(reblog)
        Fracvarcomp = integrate.simpson(psdcomp.power,psdcomp.freq)
        
        # Model PSDs to generate fake LC samples for MCMC simulations
        # Initialise fitting parameters
        pfitref = psdref.power
        perrfitref = psdref.power_err
        freqfitref = psdref.freq
        pfit = psdcomp.power
        perrfit = psdcomp.power_err
        freqfit = psdcomp.freq
    
        # Perform PSD fitting in log space and transform back
        pfitreflog = np.log10(pfitref)
        perrfitreflog = (perrfitref)/(pfitref*np.log(10))
        amp1init = pfitreflog[0]
        alphainit = 2.0
        paramsinitial = [amp1init,alphainit]
        fitobj = kmpfit.Fitter(residuals=residuals,\
                 data=(freqfitref,pfitreflog,perrfitreflog))
        fitobj.fit(params0=paramsinitial)
        chi2min = fitobj.chi2_min
        dof = fitobj.dof
        
        pfitlog = np.log10(pfit)
        perrfitlog = (perrfit)/(pfit*np.log(10))
        amp1init2 = pfitlog[0]
        alphainit2 = 2.0
        paramsinitial2 = [amp1init2,alphainit2]
        fitobj2 = kmpfit.Fitter(residuals=residuals,\
                                data=(freqfit,pfitlog,perrfitlog))
        fitobj2.fit(params0=paramsinitial2)
        chi2min2 = fitobj2.chi2_min
        dof2 = fitobj2.dof

        amp1best = fitobj.params[0]
        amp1besterr = fitobj.xerror[0]
        alphabest = fitobj.params[1]
        alphabesterr = fitobj.xerror[1]

        amp1best2 = fitobj2.params[0]
        amp1besterr2 = fitobj2.xerror[0]
        alphabest2 = fitobj2.params[1]
        alphabesterr2 = fitobj2.xerror[1]
                
        bestfitpars = [amp1best,alphabest]
        bestfitpars_comp = [amp1best2,alphabest2]
        
        fmod = np.linspace(np.min(psdcomp.freq),np.max(psdcomp.freq),\
                           1000)
        pmodref = 10**(plmod(bestfitpars,fmod))
        pmod = 10**(plmod(bestfitpars_comp,fmod))
    
        # #Plot power-spectra
        # plt.errorbar(psdcomp.freq,psdcomp.power,\
        #              yerr=psdcomp.power_err,fmt='b.')
        # plt.errorbar(psdref.freq,psdref.power,\
        #              yerr=psdref.power_err,fmt='r.')
        # plt.plot(fmod,pmodref,'k--')
        # plt.plot(fmod,pmod,'k-')
        # plt.tick_params(axis='both', which='major', labelsize=14)
        # plt.yscale("log")
        # plt.xscale("log")
        # plt.legend(loc="best")
        # plt.show()
                        
        #Compute time lags using Stingray
        csa = AveragedCrossspectrum.from_events(evcomplc, evreflc,\
        segment_size=segsizest, dt=bsize, norm="frac",\
        use_common_mean=True)
        csa = csa.rebin_log(reblog)
        freq_lag = csa.freq
        coh, coh_e = csa.coherence()
        lag, lag_e = csa.time_lag()
        lagp, lag_ep = csa.phase_lag()
                
        #Compute time lags using my method
        freqS, dfreqS, lagS, lag_eS, cohS, coh_eS =\
        time_lag_func(countcomp,countcomp_err,countref,countref_err,\
                      countcompbkg,countrefbkg,ywindowR,\
                      Nsegments,geombin,bsize,stats)
        frqS = freqS
        frq = freq_lag
            
        # lag = lag[freq_lag>=fbinmin[k]]
        # lag_e = lag_e[freq_lag>=fbinmin[k]]
        # lagS = lagS[freq_lag>=fbinmin[k]]
        # lag_eS = lag_eS[freq_lag>=fbinmin[k]]
        # lagp = lagp[freq_lag>=fbinmin[k]]
        # lag_ep = lag_ep[freq_lag>=fbinmin[k]]
        # frq = frq[freq_lag>=fbinmin[k]]
        # frqS = frqS[freqS>=fbinmin[k]]
        
        # lag = lag[frq<=fbinmax[k]]
        # lag_e = lag_e[frq<=fbinmax[k]]
        # lagS = lagS[frq<=fbinmax[k]]
        # lag_eS = lag_eS[frq<=fbinmax[k]]
        # lagp = lagp[frq<=fbinmax[k]]
        # lag_ep = lag_ep[frq<=fbinmax[k]]
        # frq = frq[frq<=fbinmax[k]]
        # frqS = frqS[frqS<=fbinmax[k]]
        # frqp = frq[frq<=fbinmax[k]]
                
        arrayslag = np.transpose(np.column_stack((lagS,lag_eS,lagp,lag_ep,\
                                 lag,lag_e,frq,frqS)))
        arrayslag = remove_nans(arrayslag)
        lagS,lag_eS,lagp,lag_ep,lag,lag_e,frq,frqS = arrayslag
        
        lagS = lagS[lag_eS>0]
        lagp = lagp[lag_eS>0]        
        lag = lag[lag_eS>0]
        lag_eS = lag_eS[lag_eS>0]
        lag_ep = lag_ep[lag_eS>0]
        lag_e = lag_e[lag_eS>0]
        frq = frq[lag_eS>0]
        frqS = frqS[lag_eS>0]
    
        for pq in range(len(lagS)):
                                                                    
            #Shift by pi 
            sigthresh = 1.8
            ediff = np.sqrt(lag_ep[pq]**2 + lag_eS[pq]**2)
            diff = (lagp[pq] - lagS[pq])/ediff 
            niter = 3
            kiter = 0
                            
            while(abs(diff)>sigthresh):
                
                diff = (lagp[pq] - lagS[pq])/ediff 
                
                if(diff<0 and abs(diff)>sigthresh):
                    lagS[pq] -= np.pi
                    
                if(diff>0 and abs(diff)>sigthresh):
                    lagS[pq] += np.pi
                
                kiter += 1
                
                if(kiter>niter or abs(diff)<sigthresh): 
                    break
            
            lagS[pq] = lagS[pq]/(2.0*np.pi*freqS[pq])
            lag_eS[pq] = lag_eS[pq]/(2.0*np.pi*freqS[pq])
                            
        # # Lag-frequency
        # plt.figure()
        # plt.errorbar(frqS,lagS,yerr=lag_eS,fmt='r.')
        # plt.errorbar(frq,lag,yerr=lag_e,fmt='b.')
        # plt.tick_params(axis='both', which='major', labelsize=14)
        # plt.xscale("log")
        # plt.show()
                
        mean_fbS = np.median(freqS) 
        mean_lagS = np.mean(lagS)
        mean_lagSerr = np.sqrt(np.sum(lag_eS**2))/len(lagS)
        mean_fb = np.mean(freq_lag)
        mean_lag = np.median(lag)
        mean_lagerr = np.sqrt(np.sum(lag_e**2))/len(lag)
                        
        enlag.append(energies)
        denlag.append(denergies)
        mlagS.append(mean_lagS)
        mlagerrS.append(mean_lagSerr)
        mlag.append(mean_lag)
        mlagerr.append(mean_lagerr)
                                     
        fakelags = []
        for z0 in range(Ntrialmcmc):
            
            freqfk, tlagfk = mcmc(bsize,telapse,segsizest,geombin,\
                                  fbinmin[k],fbinmax[k],amp1best,\
                                  alphabest,amp1best2,alphabest2,\
                                  mucomp,muref,Plotmcmc)
            tlagfk = np.array(tlagfk)
            freqfk = np.array(freqfk)
            fakelags.append(tlagfk)
                
        fakelags = np.array(fakelags)
        mufakelag.append(np.mean(fakelags))
        fakelagerr.append(np.std(fakelags))
    
    mufakelag = np.array(mufakelag)
    fakelagerr = np.array(fakelagerr)
    confint1 = mufakelag - 0.5*3*fakelagerr
    confint2 = mufakelag + 0.5*3*fakelagerr
    
    enlag = np.array(enlag)
    mlagS = np.array(mlagS)
    mlag = np.array(mlag)
    denlag = np.array(denlag)
    mlagerrS = np.array(mlagerrS)
    mlagerr = np.array(mlagerr)
                     
    lab1 = "Frequency band: (" + str(fbinmin[k]) + "-" +\
           str(fbinmax[k]) + ") Hz: stingray"
    lab2 = "Frequency band: (" + str(fbinmin[k]) + "-" +\
           str(fbinmax[k]) + ") Hz"
    subplt = int(str(len(fbinmin)) + '1' + str(k+1)) 
    fname_save = "lag_energy_obsid_" + str(obsid) + ".png"
    
    #Lag-energy
    plt.subplot(subplt)
    plt.errorbar(enlag,mlag/ks,xerr=denlag,yerr=mlagerr/ks,\
                 fmt='bo',alpha=0.5)
    plt.errorbar(enlag,mlagS/ks,xerr=denlag,yerr=mlagerrS/ks,\
                 fmt='ro',alpha=0.5,label=lab1)
    plt.ylim(-10,10)
    # plt.plot(enlag,mufakelag/ks,'k--')
    # plt.fill_between(enlag,confint1/ks,confint2/ks,alpha=0.25,\
    # label=r"3$\sigma$ confidence level [from MCMC]")
    plt.legend(loc="lower left")
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylabel("Time lag [ks]",fontsize=14)
    plt.xscale("log")
    
    if(k!=len(fbinmin)-1):
        plt.xticks([])
    if(k==len(fbinmin)-1):
        plt.xlabel("Energy [keV]",fontsize=14)
        
plt.subplots_adjust(hspace=0)
plt.show()
                                
        
        
        
    
        

    
