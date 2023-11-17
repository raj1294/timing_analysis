import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern,\
     ExpSineSquared, WhiteKernel
import matplotlib.pyplot as plt
import glob
from kapteyn import kmpfit
import warnings
warnings.filterwarnings('ignore')

day = 86400

#Read parfile and return relevant spin-parameters
def loadparfile(pfile):
    
    t0 = 0
    f0 = 0
    fdot0 = 0
    fddot0 = 0
    
    with open(pfile) as fi:
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

def loadtimfile(tfile,T0,f0,fdot0,fddot0):
    
    times,presid,dpresid,htest = [[],[],[],[]]
    
    with open(tfile) as fi:
        for line in fi:
            line = line.split()
            if(len(line)>0 and line[0]=='photon_toa'):
                
                toas = float(line[2])
                toaserr = float(line[3])
                presid.append(float(line[12]))
                htest.append(float(line[16]))
                per = (f0 + fdot0*(toas-T0)*day +\
                       0.5*fddot0*((toas-T0)*day)**2)**-1
                freq = per**-1
                dpresid.append(toaserr*freq/1e6)
                times.append(toas)
    
    return times,presid,dpresid,htest

def gpregress(times,presid,dpresid):

    #Consider test data points on which we want to generate predictions
    #For now, use a uniform grid: Need to modify this later
    N = 10000
    dim = 1
    x_star = np.linspace(np.min(times)-1.0,np.max(times)+1.0,N)
    X = times.reshape(len(times),dim)
    X_star = x_star.reshape(N,dim)
    
    #Primary kernel parameters
    lscale = 10.0
    lscale2 = 5.0
    sigf = 0.5
    sigf2 = 1.0
    sign = np.sqrt(1e-4)    
    kern = (sigf**2)*RBF(length_scale=lscale) +\
            (sigf2**2)*RBF(length_scale=lscale2) +\
            WhiteKernel(noise_level=sign) 
    
    #Alternative kernels
    # per = 50.0
    # smoothness = 2.5
    # kern = (sigf**2)*Matern(length_scale=lscale,nu=smoothness) +\
    #         WhiteKernel(noise_level=sign)
    # kern = (sigf**2)*ExpSineSquared(length_scale=lscale,periodicity=per) +\
    #         WhiteKernel(noise_level=sign)
    
    #Fit GP model
    gp = GaussianProcessRegressor(kernel=kern,alpha=1e-10,\
          n_restarts_optimizer=200,normalize_y=False)
    gp.fit(X,presid)
    score = gp.score(X,presid)
    params = gp.kernel_
    print(params)
    print(score)

    # lscale = np.logspace(-3, 3, num=400)
    # sigmaf = np.logspace(-2, 3, num=400)
    # lscale_grid, sigmaf_grid = np.meshgrid(lscale, sigmaf)
    # neval = np.sqrt(5.9e-4)
    
    # LGP = [gp.log_marginal_likelihood(theta=np.log([neval, scale, noise]))
    #         for scale, noise in zip(lscale_grid.ravel(),sigmaf_grid.ravel())]
    # LGP = np.reshape(LGP, newshape=sigmaf_grid.shape)
    
    # likelihoodGP = np.transpose(LGP)
    # lscaleposterior = []
    # for k0 in range(len(likelihoodGP)):
    #     lscaleposterior.append(integrate.simps(likelihoodGP[k0],sigmaf))
    # lscaleposterior = np.array(lscaleposterior)
    # normlscale = integrate.simps(lscaleposterior,lscale)
    # lscaleposterior/=normlscale
    
    # likelihoodGP = np.transpose(likelihoodGP)
    # sigfposterior = []
    # for k1 in range(len(likelihoodGP)):
    #     sigfposterior.append(integrate.simps(likelihoodGP[k1],lscale))
    # sigfposterior = np.array(sigfposterior)
    # normsigf = integrate.simps(sigfposterior,sigmaf)
    # sigfposterior/=normsigf
            
    # fig = plt.figure()
    # ax0 = fig.add_axes([.2, .7, .4, .2])
    # ax0.plot(lscale,-lscaleposterior,'r-')
    # ax0.tick_params(axis='both',which='major',labelsize=14)
    # ax0.set_xscale("log")
    # ax0.set_xlabel("Length-scale [days]",fontsize=14)
    # ax0.set_ylabel("PDF",fontsize=14)
    
    # ax1 = fig.add_axes([.7, .1, .2, .5])
    # ax1.plot(-sigfposterior,sigmaf,'b-')
    # ax1.tick_params(axis='both',which='major',labelsize=14)
    # ax1.set_yscale("log")
    # ax1.set_ylabel(r"$\sigma_{f}$",fontsize=14)
    # ax1.set_xlabel("PDF",fontsize=14)
    
    # ax2 = fig.add_axes([.2, .1, .4, .5])
    # im = ax2.pcolormesh(lscale_grid,sigmaf_grid,likelihoodGP,\
    #                     vmin=-140,vmax=-40)
    # cbar = fig.colorbar(im,orientation='vertical')
    # cbar.set_label(r'Likelihood',fontsize=14)
    # cbar.ax.tick_params(labelsize=14)

    # plt.xscale("log")
    # plt.yscale("log")
    # plt.xlabel("Length scale [days]",fontsize=14)
    # plt.ylabel("Noise-level",fontsize=14)
    # plt.show()
    
    #Best-fit prediction
    y_pred, dy_pred = gp.predict(X_star,return_std=True)
        
    return x_star, y_pred, dy_pred

def spinpar(x_star,y_pred,dy_pred,t0,f0,fdot0,fddot0,siguncer):

    dx = (x_star[1]-x_star[0])*day
    y_predmin = y_pred - 0.5*siguncer*dy_pred
    y_predmax = y_pred + 0.5*siguncer*dy_pred

    grad = np.gradient(y_pred, dx)
    graddown = np.gradient(y_predmin, dx)
    gradup = np.gradient(y_predmax, dx)
    
    freq = f0 + fdot0*((x_star-t0)*day) +\
           0.5*(fddot0)*((x_star-t0)*day)**2 - grad
    freqdown = f0 + fdot0*((x_star-t0)*day) +\
                0.5*(fddot0)*((x_star-t0)*day)**2 - graddown
    frequp = f0 + fdot0*((x_star-t0)*day) +\
              0.5*(fddot0)*((x_star-t0)*day)**2 - gradup
    df = 0.5*(frequp-freqdown)
    
    fdot = fdot0 + (fddot0)*((x_star-t0)*day) - np.gradient(grad, dx)
    fdotdown = fdot0 + 0.5*(fddot0)*((x_star-t0)*day)**2 -\
                np.gradient(graddown, dx)
    fdotup = fdot0 + 0.5*(fddot0)*((x_star-t0)*day)**2 -\
              np.gradient(gradup, dx)
    dfdot = 0.5*(fdotup-fdotdown)
    
    fddot = -np.gradient(fdot, dx)
    fddotdown = -np.gradient(fdotdown, dx)
    fddotup = -np.gradient(fdotup, dx)
    dfddot = 0.5*(fddotup-fddotdown)
    
    return x_star,freq,df,fdot,dfdot,fddot,dfddot

def findcloseparams(tinterp,x_star,pmod,dpmod,fmod,dfmod,fdotmod,\
                           dfdotmod,fddotmod,dfddotmod):

    tpred = np.zeros(len(tinterp))
    ppred = np.zeros(len(tinterp))
    dppred = np.zeros(len(tinterp))
    fpred = np.zeros(len(tinterp))
    dfpred = np.zeros(len(tinterp))
    fdotpred = np.zeros(len(tinterp))
    dfdotpred = np.zeros(len(tinterp))
    fddotpred = np.zeros(len(tinterp))
    dfddotpred = np.zeros(len(tinterp))

    for j in range(len(tinterp)):
        mynum = min(x_star, key=lambda x:abs(x-tinterp[j]))
        xlist = list(x_star)
        mynumind = xlist.index(mynum)
        tpred[j] = mynum
        ppred[j] = pmod[mynumind]
        dppred[j] = dpmod[mynumind]
        fpred[j] = fmod[mynumind]
        dfpred[j] = dfmod[mynumind]
        fdotpred[j] = fdotmod[mynumind]
        dfdotpred[j] = dfdotmod[mynumind]
        fddotpred[j] = fddotmod[mynumind]
        dfddotpred[j] = dfddotmod[mynumind]

    return tpred,ppred,dppred,fpred,dfpred,fdotpred,dfdotpred,\
           fddotpred,dfddotpred

def interp(tinterp,x_star,pmod,dpmod,fmod,dfmod,fdotmod,dfdotmod,\
           fddotmod,dfddotmod):
    
    tpred,ppred,dppred,fpred,dfpred,fdotpred,dfdotpred,fddotpred,dfddotpred =\
    findcloseparams(tinterp,x_star,pmod,dpmod,fmod,dfmod,fdotmod,dfdotmod,\
                   fddotmod,dfddotmod)
    
    fgrad = fpred - (f0 + fdot0*((tpred-t0)*day) +\
            0.5*(fddot0)*((tpred-t0)*day)**2)
    fdotgrad = fdotpred - (fdot0 + (fddot0)*((tpred-t0)*day))
    fddotgrad = fddotpred - fddot0
    
    pmodinterp = ppred + fgrad*(tinterp-tpred)*day +\
                 0.5*fdotgrad*((tinterp-tpred)*day)**2 +\
                 (1./6.)*fddotgrad*((tinterp-tpred)*day)**3
    finterp = fpred + fdotpred*(tinterp-tpred)*day +\
              0.5*fddotpred*((tinterp-tpred)*day)**2
    fdotinterp = fdotpred + fddotpred*((tinterp-tpred)*day)
    
    dpmodinterp = dppred**2 + (dfpred**2)*((tinterp-tpred)*day) +\
                  0.5*(dfdotpred**2)*(((tinterp-tpred)*day)**2) +\
                  (1./6.)*(dfddotpred**2)*((tinterp-tpred)*day)**3
    dfinterp = (dfpred**2 + ((tinterp-tpred)*day)*(dfdotpred**2) +\
               (dfdotpred**2)*(0.5*((tinterp-tpred)*day)**2))**0.5
    dfdotinterp = ((dfdotpred**2) + ((tinterp-tpred)*day)*dfddotpred**2)**0.5
        
    return tinterp,pmodinterp,dpmodinterp,finterp,dfinterp,\
           fdotinterp,dfdotinterp,fddotpred,dfddotpred

def rungpregress(binary,times,phase_resid,dphase_resid,fout):
    if(np.int64(binary)==1):
        x_star, y_pred, dy_pred = gpregress(times,phase_resid,dphase_resid)
        X = np.column_stack((x_star,y_pred,dy_pred))
        np.savetxt(fout,X,fmt="%s",delimiter="   ")

##############################################################################
##############################################################################

#Save timing solutions
#Segment 1
tpre = [58906.9256518327400500,58907.1255731793046952,\
        58934.6068852060213210,58935.7131264931486095,\
        58936.1724486628724043,58937.8447966938036195,\
        58938.0385094903200084,58941.6547053934028226,\
        58942.2418360070980995,58968.7077145224700699,\
        58996.6640612751510244,59026.1839643124294272]

tpre0 = 58968.65257659003
fpre0 = 3.038460778559553982
dfpre0 = 5.918641102011889866e-09
fdotpre0 = -6.599902400624579366e-11
dfdotpre0 = 1.4085514864038731471e-15
fddotpre0 = 9.961751224505475761e-21
dfddotpre0 = 1.5445902749445420205e-21
freqpre = np.zeros(len(tpre))
dfreqpre = np.zeros(len(tpre))
fdotpre = np.zeros(len(tpre))
dfdotpre = np.zeros(len(tpre))
for k in range(len(tpre)):
    freqpre[k] = fpre0 + fdotpre0*((tpre[k]-tpre0)*day) +\
              0.5*fddotpre0*((tpre[k]-tpre0)*day)**2
    dfreqpre[k] = np.sqrt(dfpre0**2+(dfdotpre0**2)*((tpre[k]-tpre0)*day)**2 +\
                  (1./4.)*(dfddotpre0**2)*((tpre[k]-tpre0)*day)**4) 
for k2 in range(len(tpre)):
    fdotpre[k2] = fdotpre0 + fddotpre0*((tpre[k2]-tpre0)*day)
    dfdotpre[k2] = np.sqrt((dfdotpre0**2) + (dfddotpre0**2)*\
                        ((tpre[k2]-tpre0)*day)**2 )
fdotpre/=1e-11
dfdotpre/=1e-11

#Segment 4
tpost = [59281.3234820961184040,59282.3615208860981304,\
          59284.4201632377128171,59312.3276851932492021,\
          59318.3902132661846891,59319.7639154593673411,\
          59321.8860030377648040,\
          59322.5306829251902548,59337.5726974733481727,\
          59338.0903543255330898,59373.8634868290559976,\
          59409.7861215255505987,\
          59410.1099776043444343,59417.9791930818282606,\
          59418.2374832283229402,59434.9130316952477407]
tpost = np.array(tpost)

tpost0 = 59318.218098148460
fpost0 = 3.0364318524107750427
dfpost0 = 5.2095403062190489465e-09
fdotpost0 = -6.70858832634465154e-11
dfdotpost0 = 3.4929208588650526122e-15
fddotpost0 = 8.950390183552349445e-22
dfddotpost0 = 1.1691309533299058355e-21
freqpost = fpost0 + fdotpost0*((tpost-tpost0)*day) +\
            0.5*fddotpost0*((tpost-tpost0)*day)**2
dfreqpost = np.sqrt(dfpost0**2 + (dfdotpost0**2)*((tpost-tpost0)*day)**2 +\
                    (0.5*(dfddotpost0)*((tpost-tpost0)*day)**2)**2)
fdotpost = fdotpost0 + fddotpost0*(tpost-tpost0)*day 
dfdotpost = np.sqrt(dfdotpost0**2 + dfddotpost0**2*((tpost-tpost0)*day)**2)
fdotpost/=1e-11
dfdotpost/=1e-11

toutburst = 59062.8423169 + np.zeros(100)
ytoutburst = np.linspace(3.02,3.06,100)
ytoutburst2 = np.linspace(-7.8,-6.3,100)
ytoutburst3 = np.linspace(-10,100,100)

parfile = 'post_outburst.par'
t0,f0,fdot0,fddot0 = loadparfile(parfile)

toas,presid,dpresid,htest = np.loadtxt("phase_resid_clean.dat",skiprows=0,\
                                        unpack=True)
rungp = np.int64(0)
fout = "GPmod.dat"

rungpregress(rungp,toas,presid,dpresid,fout)
x_star, y_pred, dy_pred = np.loadtxt(fout,skiprows=0,unpack=True)
        
#Compute spin parameters and errors based on best-fit GP prediction
siguncer = 3.0 #Number of uncertainty std deviations
tmod,fmod,dfmod,fdotmod,dfdotmod,fddotmod,dfddotmod =\
spinpar(x_star,y_pred,dy_pred,t0,f0,fdot0,fddot0,siguncer)

#Interpolate 
toas,ymodinterp,dymodinterp,fpred,dfpred,fdotpred,dfdotpred,\
    fddotpred,dfddotpred = interp(toas,x_star,y_pred,dy_pred,\
    fmod,dfmod,fdotmod,dfdotmod,fddotmod,dfddotmod)

#Compute chi^2 and degrees of freedom
resid = (presid - ymodinterp)/dpresid
residerr = np.ones(len(resid))
chi2 = np.sum((resid)**2)
dof = len(resid)
print(chi2,dof)

# #Filter out data that lie outside the 3 sigma interval
# rthresh = 2.2
# toas = toas[np.abs(resid)<rthresh]
# presid = presid[np.abs(resid)<rthresh]
# dpresid = dpresid[np.abs(resid)<rthresh]
# fpred = fpred[np.abs(resid)<rthresh]
# dfpred = dfpred[np.abs(resid)<rthresh]
# fdotpred = fdotpred[np.abs(resid)<rthresh]
# dfdotpred = dfdotpred[np.abs(resid)<rthresh]
# htest = htest[np.abs(resid)<rthresh]
# residerr = residerr[np.abs(resid)<rthresh]
# resid = resid[np.abs(resid)<rthresh]
# chi2 = np.sum((resid)**2)
# dof = len(resid)
# print(chi2,dof)
# W = np.column_stack((toas,presid,dpresid,htest))
# np.savetxt("phase_resid_clean.dat",W,fmt='%s',delimiter='   ')

#Spin parameters (numerical)
# Obtain spin parameters and separate data associated with different 
# instruments
toas_nicer,presid_nicer,dpresid_nicer,htest_nicer =\
    np.loadtxt("phase_resid_nicer.dat",skiprows=0,unpack=True)
toas_nustar,presid_nustar,dpresid_nustar,htest_nustar =\
    np.loadtxt("phase_resid_nustar.dat",skiprows=0,unpack=True)
toas_xmm,presid_xmm,dpresid_xmm,htest_xmm =\
    np.loadtxt("phase_resid_xmm.dat",skiprows=0,unpack=True)
toas_swift,presid_swift,dpresid_swift,htest_swift =\
    np.loadtxt("phase_resid_swift.dat",skiprows=0,unpack=True)

resid_nicer,resid_nustar,resid_xmm,resid_swift = [[],[],[],[]]
dresid_nicer,dresid_nustar,dresid_xmm,dresid_swift = [[],[],[],[]]
fnicer,fnustar,fxmm,fswift = [[],[],[],[]]
fbasenicer,fbasenustar,fbasexmm,fbaseswift = [[],[],[],[]]
fdotnicer,fdotnustar,fdotxmm,fdotswift = [[],[],[],[]]
dfnicer,dfnustar,dfxmm,dfswift = [[],[],[],[]]
dfdotnicer,dfdotnustar,dfdotxmm,dfdotswift = [[],[],[],[]]
newtoa_nicer,newtoa_nustar,newtoa_xmm,newtoa_swift = [[],[],[],[]]

for k in range(len(toas)):
    
    for num1 in range(len(toas_nicer)):
        if(toas_nicer[num1]==toas[k]):
            
            newtoa_nicer.append(toas[k])
            fnicer.append(fpred[k])
            fdotnicer.append(fdotpred[k]/1e-11)
            dfnicer.append(dfpred[k])
            dfdotnicer.append(dfdotpred[k]/1e-11)
    
    for num2 in range(len(toas_nustar)):
        if(toas_nustar[num2]==toas[k]):
            
            newtoa_nustar.append(toas[k])
            fnustar.append(fpred[k])
            fdotnustar.append(fdotpred[k]/1e-11)
            dfnustar.append(dfpred[k])
            dfdotnustar.append(dfdotpred[k]/1e-11)

    for num3 in range(len(toas_swift)):
        if(toas_swift[num3]==toas[k]):
            
            newtoa_swift.append(toas[k])
            fswift.append(fpred[k])
            fdotswift.append(fdotpred[k]/1e-11)
            dfswift.append(dfpred[k])
            dfdotswift.append(dfdotpred[k]/1e-11)

    for num4 in range(len(toas_xmm)):
        if(toas_xmm[num4]==toas[k]):
            
            newtoa_xmm.append(toas[k])
            fxmm.append(fpred[k])
            fdotxmm.append(fdotpred[k]/1e-11)
            dfxmm.append(dfpred[k])
            dfdotxmm.append(dfdotpred[k]/1e-11)

#Leave out data close to boundaries
newtoa_nicer = newtoa_nicer[0:-5]
fnicer = fnicer[0:-5]
dfnicer = dfnicer[0:-5]
fdotnicer = fdotnicer[0:-5]
dfdotnicer = dfdotnicer[0:-5]

#Fluxes
ftoanicer,fluxnicer,dfluxnicer = [[],[],[]] #NICER
for file in glob.glob("flux*.dat"):
    ts,flx,dflx = np.loadtxt(file,skiprows=0,unpack=True)
    ts = list(ts)
    flx = list(flx)
    dflx = list(dflx)
    ftoanicer += ts
    fluxnicer += flx
    dfluxnicer += dflx
#Remaining instruments
timesxmm = [59109.74949092488,59124.88388268009,59146.08058052532]
pfluxxmm = [10**-11.0983,10**-11.1389,10**-11.1620]
dpfluxxmm = [9.2e-13,6.69e-13,4.75e-13]
pfluxxmm = np.array(pfluxxmm)
dpfluxxmm = np.array(dpfluxxmm)
pfluxxmm/=1e-11
dpfluxxmm/=1e-11

#Figure 1 of paper
plt.figure()
plt.subplot(311)
plt.plot(toutburst,ytoutburst3,'k-',label="Swift-BAT detection of ms burst")
plt.fill_betweenx(y=[-2,10],x1= [58000,58000], x2= [59055,59055],\
                  alpha=0.05,color="blue")
plt.fill_betweenx(y=[-8,100],x1= [59055,59055], x2= [59260,59260],\
                  alpha=0.05,color="darkorange")
plt.fill_betweenx(y=[-8,100],x1= [59260,59260], x2= [59550,59550],\
                  alpha=0.05,color="red")
plt.errorbar(ftoanicer,fluxnicer,yerr=dfluxnicer,fmt='b.')
plt.errorbar(timesxmm,pfluxxmm,yerr=dpfluxxmm,fmt='r.')
# plt.errorbar(timesswift,pfluxswift,yerr=dpfluxswift,fmt='k.',label="Swift")
# plt.errorbar(timesnustar,pfluxnustar,yerr=dpfluxnustar,fmt='g.',label="NuSTAR")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.text(58360,8.5,"Segment 1",ha='left',va='center',fontsize=12,\
          bbox={'ec':'k','fill':False})
plt.text(59080,8.5,"Segment 2",ha='left',va='center',fontsize=12,\
          bbox={'ec':'k','fill':False})
plt.text(59350,8.5,"Segment 3",ha='left',va='center',fontsize=12,\
          bbox={'ec':'k','fill':False})
plt.xticks([])
plt.ylim(0,5)
plt.ylabel(r"Flux [10$^{-11}$ erg cm$^{-2}$ s$^{-1}$]",fontsize=12)
plt.legend(loc="best")
plt.xlim(58880,59450)

plt.subplot(312)
plt.plot(toutburst,ytoutburst,'k-')
plt.errorbar(tpre,freqpre,yerr=dfreqpre,fmt='b.')
plt.errorbar(tpost,freqpost,yerr=dfreqpost,fmt='b.')
plt.errorbar(newtoa_nicer,fnicer,yerr=dfnicer,fmt='b.',label="NICER")
plt.errorbar(newtoa_xmm,fxmm,yerr=dfxmm,fmt='r.',label="XMM-Newton")
plt.errorbar(newtoa_nustar,fnustar,yerr=dfnustar,fmt='g.',label="NuSTAR")
plt.errorbar(newtoa_swift,fswift,yerr=dfswift,fmt='k.',label="Swift")
plt.ticklabel_format(useOffset=False)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xticks([])
plt.fill_betweenx(y=[-8,6],x1= [58000,58000], x2= [59055,59055],\
                  alpha=0.05,color="blue")
plt.fill_betweenx(y=[-8,100],x1= [59055,59055], x2= [59260,59260],\
                  alpha=0.05,color="darkorange")
plt.fill_betweenx(y=[-8,100],x1= [59260,59260], x2= [59550,59550],\
                  alpha=0.05,color="red")
plt.legend(loc="best")
plt.ylabel(r"$\nu$ [Hz]",fontsize=12)
plt.xlim(58880,59450)
plt.ylim(3.034,3.044)

plt.subplot(313)
plt.fill_betweenx(y=[-8,6],x1= [58000,58000], x2= [59055,59055],\
                  alpha=0.05,color="blue")
plt.fill_betweenx(y=[-8,10],x1= [59055,59055], x2= [59260,59260],\
                  alpha=0.05,color="darkorange")
plt.fill_betweenx(y=[-8,100],x1= [59260,59260], x2= [59550,59550],\
                  alpha=0.05,color="red")
plt.plot(toutburst,ytoutburst2,'k-',label="Swift-BAT detection of ms burst")
plt.errorbar(tpre,fdotpre,yerr=dfdotpre,fmt='b.')
plt.errorbar(tpost,fdotpost,yerr=dfdotpost,fmt='b.')
plt.errorbar(newtoa_nicer,fdotnicer,yerr=dfdotnicer,fmt='b.',label="NICER")
plt.errorbar(newtoa_xmm,fdotxmm,yerr=dfdotxmm,fmt='r.',label="XMM-Newton")
plt.errorbar(newtoa_nustar,fdotnustar,yerr=dfdotnustar,fmt='g.',label="NuSTAR")
plt.errorbar(newtoa_swift,fdotswift,yerr=dfdotswift,fmt='k.',label="Swift")
plt.ylabel(r"$\dot{\nu}$ [10$^{-11}$ Hz/s]",fontsize=12)
plt.xlabel("Time [MJD]",fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlim(58880,59450)
plt.ylim(-7.4,-6.4)
plt.subplots_adjust(hspace=0)
plt.show()

#Pulsed flux vs nu-dot
pf,pferr,nudot,nudoterr = [[],[],[],[]]
for k in range(len(ftoanicer)):
    for k2 in range(len(newtoa_nicer)):
        if(abs(newtoa_nicer[k2]-ftoanicer[k])<0.25):
            pf.append(fluxnicer[k])
            pferr.append(dfluxnicer[k])
            nudot.append(fdotnicer[k2])
            nudoterr.append(dfdotnicer[k2])

plt.figure()
plt.errorbar(nudot,pf,xerr=nudoterr,yerr=pferr,fmt='b.',label="NICER")
plt.xlabel(r"$\dot{\nu}$ [10$^{-11}$ Hz/s]",fontsize=12)
plt.ylabel(r"Flux [10$^{-11}$ erg cm$^{-2}$ s$^{-1}$]",fontsize=12)
plt.legend(loc="best")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()

ftimesol = 'timesol_post_nicer.dat'
W = np.column_stack((newtoa_nicer,fnicer,dfnicer,fdotnicer,dfdotnicer))
np.savetxt(ftimesol,W,fmt='%s',delimiter='   ')

ftimesol = 'timesol_post_nustar.dat'
X = np.column_stack((newtoa_nustar,fnustar,dfnustar,fdotnustar,dfdotnustar))
np.savetxt(ftimesol,X,fmt='%s',delimiter='   ')

ftimesol = 'timesol_post_xmm.dat'
Y = np.column_stack((newtoa_xmm,fxmm,dfxmm,fdotxmm,dfdotxmm))
np.savetxt(ftimesol,Y,fmt='%s',delimiter='   ')

ftimesol = 'timesol_post_swift.dat'
Z = np.column_stack((newtoa_swift,fswift,dfswift,fdotswift,dfdotswift))
np.savetxt(ftimesol,Z,fmt='%s',delimiter='   ')

#Timing residulas with overlaid best-fit model
plt.figure()
plt.subplot(211)
plt.errorbar(toas,presid,yerr=dpresid,fmt='ko',\
              label="Residuals: all instruments")
plt.title("Segment 2",fontsize=12)
plt.plot(x_star, y_pred, 'b--', label='Best-fit GP model')
plt.legend(loc="best")
plt.subplot(212)
plt.errorbar(toas,resid,yerr=residerr,fmt='ko')
plt.xlabel("Time [MJD]",fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylabel(r"Pre-fit residuals [cycles]",fontsize=12)
plt.subplots_adjust(hspace=0)
plt.show()

#Comparison with segmented approach
pepoch_em,_,_,f0em,df0em,f1em,df1em = np.loadtxt("segments_F0_F1.txt",\
                                                  skiprows=1,unpack=True)
fb_em = f0 + fdot0*((pepoch_em-t0)*day) + 0.5*fddot0*((pepoch_em-t0)*day)**2        
f0em = f0em - fb_em
pepoch_em = pepoch_em[8:-3]
f0em = f0em[8:-3]
df0em = df0em[8:-3]
f1em = f1em[8:-3]
df1em = df1em[8:-3]
pepoch_em = np.array(pepoch_em)
f0em = np.array(f0em)
f0em += f0 + fdot0*(pepoch_em-t0)*day

#Hu et al. 2023
toashu = [59064.0,59099.0,59145.0]
freqhu = [3.037919719,3.037707087,3.037432343]
dfreqhu = [23e-9,50e-9,15e-9]
fdothu = [-6.7256,-7.0180,-6.8498]
dfdothu = [25e-4,52e-4,12e-4]

tcomp = np.linspace(59054,59165,100)
fdotcomp = -6.599902400624579366 + np.zeros(100)
fdotcomp2 = -6.70858832634465154 + np.zeros(100)

#Save f-dots
times = np.hstack((newtoa_nicer,newtoa_xmm,newtoa_nustar,newtoa_swift))
freqs = np.hstack((fnicer,fxmm,fnustar,fswift))
dfreqs = np.hstack((dfnicer,dfxmm,dfnustar,dfswift))
fdots = np.hstack((fdotnicer,fdotxmm,fdotnustar,fdotswift))
dfdots = np.hstack((dfdotnicer,dfdotxmm,dfdotnustar,dfdotswift))
times,freqs,dfreqs,fdots,dfdots = \
                            zip(*sorted(zip(times,freqs,dfreqs,fdots,dfdots)))
Q = np.column_stack((times,fdots,dfdots))
np.savetxt("fdot_evol.dat",Q,fmt='%s',delimiter='   ')

plt.figure()
plt.subplot(121)
plt.errorbar(pepoch_em,f0em,yerr=df0em,fmt='k.',\
              label="Classic segmented approach")
plt.errorbar(toashu,freqhu,yerr=dfreqhu,fmt='rD',label="Hu et al. 2023")
plt.errorbar(times,freqs,yerr=dfreqs,fmt='g.',label="GP regression")
plt.ticklabel_format(useOffset=False)
plt.legend(loc="best")
plt.ylabel(r"$\nu$ [Hz]",fontsize=12)
plt.xlabel("Time [MJD]",fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlim(59054,59165)
plt.subplot(122)
plt.errorbar(toashu,fdothu,yerr=dfdothu,fmt='rD',label="Hu et al. 2023")
plt.plot(tcomp,fdotcomp,'k--',alpha=0.5,label="Pre-outburst solution")
plt.plot(tcomp,fdotcomp2,'m--',alpha=0.5,label="Post-outburst solution")
plt.xlim(59054,59165)
plt.errorbar(pepoch_em,f1em/1e-11,yerr=df1em/1e-11,fmt='k.',\
              label="Classic segmented approach")
plt.plot(times,fdots,'g.',label="GP regression")
plt.xlabel("Time [MJD]",fontsize=12)
plt.ylabel(r"$\dot{\nu}$ [10$^{-11}$ Hz s$^{-1}$]",fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(loc="best")
plt.show()