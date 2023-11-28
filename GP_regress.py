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
    N = 10000
    dim = 1
    x_star = np.linspace(np.min(times)-1.0,np.max(times)+1.0,N)
    X = times.reshape(len(times),dim)
    X_star = x_star.reshape(N,dim)
    
    #Primary kernel parameters (RBF)
    lscale = 10.0
    lscale2 = 5.0
    sigf = 0.5
    sigf2 = 1.0
    sign = np.sqrt(1e-4)    
    kern = (sigf**2)*RBF(length_scale=lscale) +\
            (sigf2**2)*RBF(length_scale=lscale2) +\
            WhiteKernel(noise_level=sign) 
    
    #Alternative kernels (Matern and exponential sine squared)
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

    lscale = np.logspace(-3, 3, num=400)
    sigmaf = np.logspace(-2, 3, num=400)
    lscale_grid, sigmaf_grid = np.meshgrid(lscale, sigmaf)
    neval = np.sqrt(5.9e-4)
    
    LGP = [gp.log_marginal_likelihood(theta=np.log([neval, scale, noise]))
            for scale, noise in zip(lscale_grid.ravel(),sigmaf_grid.ravel())]
    LGP = np.reshape(LGP, newshape=sigmaf_grid.shape)
    
    likelihoodGP = np.transpose(LGP)
    lscaleposterior = []
    for k0 in range(len(likelihoodGP)):
        lscaleposterior.append(integrate.simps(likelihoodGP[k0],sigmaf))
    lscaleposterior = np.array(lscaleposterior)
    normlscale = integrate.simps(lscaleposterior,lscale)
    lscaleposterior/=normlscale
    
    likelihoodGP = np.transpose(likelihoodGP)
    sigfposterior = []
    for k1 in range(len(likelihoodGP)):
        sigfposterior.append(integrate.simps(likelihoodGP[k1],lscale))
    sigfposterior = np.array(sigfposterior)
    normsigf = integrate.simps(sigfposterior,sigmaf)
    sigfposterior/=normsigf
            
    fig = plt.figure()
    ax0 = fig.add_axes([.2, .7, .4, .2])
    ax0.plot(lscale,-lscaleposterior,'r-')
    ax0.tick_params(axis='both',which='major',labelsize=14)
    ax0.set_xscale("log")
    ax0.set_xlabel("Length-scale [days]",fontsize=14)
    ax0.set_ylabel("PDF",fontsize=14)
    
    ax1 = fig.add_axes([.7, .1, .2, .5])
    ax1.plot(-sigfposterior,sigmaf,'b-')
    ax1.tick_params(axis='both',which='major',labelsize=14)
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$\sigma_{f}$",fontsize=14)
    ax1.set_xlabel("PDF",fontsize=14)
    
    ax2 = fig.add_axes([.2, .1, .4, .5])
    im = ax2.pcolormesh(lscale_grid,sigmaf_grid,likelihoodGP,\
                        vmin=-140,vmax=-40)
    cbar = fig.colorbar(im,orientation='vertical')
    cbar.set_label(r'Likelihood',fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Length scale [days]",fontsize=14)
    plt.ylabel("Noise-level",fontsize=14)
    plt.show()
    
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

parfile = 'post_outburst.par'
t0,f0,fdot0,fddot0 = loadparfile(parfile)

residual_file = "phase_resid.dat"
toas,presid,dpresid,htest = np.loadtxt(residual_file,skiprows=0,unpack=True)
                                        
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

#Filter out data that lie outside the 3 sigma interval
rthresh = 3.0
toas = toas[np.abs(resid)<rthresh]
presid = presid[np.abs(resid)<rthresh]
dpresid = dpresid[np.abs(resid)<rthresh]
fpred = fpred[np.abs(resid)<rthresh]
dfpred = dfpred[np.abs(resid)<rthresh]
fdotpred = fdotpred[np.abs(resid)<rthresh]
dfdotpred = dfdotpred[np.abs(resid)<rthresh]
htest = htest[np.abs(resid)<rthresh]
residerr = residerr[np.abs(resid)<rthresh]
resid = resid[np.abs(resid)<rthresh]
chi2 = np.sum((resid)**2)
dof = len(resid)
print(chi2,dof)

phase_new = "phase_resid_clean.dat"
W = np.column_stack((toas,presid,dpresid,htest))
np.savetxt(phase_new,W,fmt='%s',delimiter='   ')
