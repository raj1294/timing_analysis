import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob,re

day = 86400

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

parfile = "post_outburst.par"
T0,f0,fdot0,fddot0 = loadparfile(parfile)
times,presid,dpresid,htest = np.loadtxt("phase_resid_clean.dat",skiprows=0,\
                                        unpack=True)
times = list(times)
presid = list(presid)
dpresid = list(dpresid)
htest = list(htest)    
times,presid,dpresid,htest = zip(*sorted(zip(times,presid,dpresid,htest)))
times = np.array(times)
presid = np.array(presid)
dpresid = np.array(dpresid)
htest = np.array(htest)

#Optimised hyperparameters
lscale = 16*day
sigf = np.sqrt(5.9)
sign = np.sqrt(6.14e-4)

#Squared-exponential plus white noise kernel
trainpoints = times
Kij = np.zeros((len(trainpoints),len(trainpoints)))
for i in range(len(trainpoints)):
    for j in range(len(trainpoints)):
        dist = abs((trainpoints[i] - trainpoints[j])*day)
        Kij[i][j] = (sigf**2)*np.exp(-(dist**2)/(2.0*lscale**2))
        if(i==j):
            Kij[i][j] += sign**2 
Kij = np.matrix(Kij)
Kijinv = np.linalg.inv(Kij)

#Matrix of test predictions
testpoints = times
Kstar = np.zeros((len(testpoints),len(trainpoints)))
for i1 in range(len(testpoints)):
    for j1 in range(len(trainpoints)):
        dist = abs((testpoints[i1] - trainpoints[j1])*day)
        Kstar[i1][j1] = (sigf**2)*np.exp(-(dist**2)/(2.0*lscale**2))
Kstar = np.matrix(Kstar)
y = np.transpose(np.matrix(presid))
ystar = np.matmul(np.matmul(Kstar,Kijinv),y)
ystar = np.array(np.transpose(ystar))[0]

#Covariance matrix of test predictions 
Kstarstar = np.zeros((len(testpoints),len(testpoints)))
for i2 in range(len(testpoints)):
    for j2 in range(len(testpoints)):
        dist = abs((testpoints[i2] - testpoints[j2])*day)
        Kstarstar[i2][j2] = (sigf**2)*np.exp(-(dist**2)/(2.0*lscale**2))
Kstarstar = np.matrix(Kstarstar)
varystar = Kstarstar - np.matmul(np.matmul(Kstar,Kijinv),np.transpose(Kstar))
varystar = np.abs(np.diag(varystar))
sigystar = 3.0*np.sqrt(varystar)

#Spin parameter predictions
Kstarprime = np.zeros((len(testpoints),len(trainpoints)))
Kstardprime = np.zeros((len(testpoints),len(trainpoints)))
for i3 in range(len(testpoints)):
    for j3 in range(len(trainpoints)):
        dist = ((testpoints[i3] - trainpoints[j3])*day)
        Kstarprime[i3][j3] = (dist*(sigf/lscale)**2)*\
                              np.exp(-0.5*(dist/lscale)**2)
        Kstardprime[i3][j3] = ((1-(dist/lscale)**2)*(sigf/lscale)**2)*\
                              np.exp(-0.5*(dist/lscale)**2)
Kstarprime = np.matrix(Kstarprime)
Kstardprime = np.matrix(Kstardprime)

fstar = np.matmul(np.matmul(Kstarprime,Kijinv),y)
P = np.matmul(np.matmul(Kstarprime,Kijinv),np.transpose(Kstarprime))
vfstar = np.abs(-np.diag(P) + (sigf/lscale)**2)
vfstar = np.sqrt(vfstar)

fdotstar = np.matmul(np.matmul(Kstardprime,Kijinv),y)
Q = np.matmul(np.matmul(Kstardprime,Kijinv),np.transpose(Kstardprime))
vfdotstar = np.abs(-np.diag(Q) + 3.0*(sigf**2)/(lscale**4))
vfdotstar = np.sqrt(vfdotstar)

#Convert to arrays
fstar = np.array(np.transpose(fstar))[0]
fdotstar = np.array(np.transpose(fdotstar))[0]
vfstar = np.array(vfstar)
vfdotstar = np.array(vfdotstar)

fbase = f0 + fdot0*((testpoints-T0)*day) + 0.5*fddot0*((testpoints-T0)*day)**2        
fdotbase = fdot0 + fddot0*((testpoints-T0)*day)
fdotstar = fdotbase + fdotstar
fstar += fbase

#Pre-outburst solution
tpre = [58351.4401043439694958,58355.7535225175590112,\
        58421.2874969483058292,58552.2294539096939822,\
        58553.2102962967427730,58555.3244606205218331,\
        58559.5084409513906544,58609.7651860492967278,\
        58609.9535910265855376,58640.2715363813537360,\
        58671.1550720921811162,58698.2911569776422309,\
        58729.7022350232772173,58750.7911535246184459,\
        58757.4023851791734618,58763.8483410002757923,\
        58765.7757425068161288,58765.9046764251591117,\
        58769.2081406626138528,58934.6068879675919292,\
        58935.9634726495298094,58937.8437673092541498,\
        58938.0380818872171365,58941.6531312807656143,\
        58942.2418387231667676,58968.5141647359088606,\
        58968.7077211746691909,58968.8463490872098900,\
        58996.7399512719712946]
tpre = np.array(tpre)

tpre0 = 58660.0900280
fpre0 = 3.0402222872487625
dfpre0 = 4.3099511233366853074e-09
fdotpre0 = -6.612534352391e-11
dfdotpre0 = 4.3416643759482977115e-16
fddotpre0 = 3.225658090165e-21
dfddotpre0 = 1.1821402177936449526e-23
f3pre0 = 2.358775115805e-29
df3pre0 = 3.7867145112703236896e-30
tg = 58661.0
df0g = 1.3090224e-07

freqpre = np.zeros(len(tpre))
dfreqpre = np.zeros(len(tpre))
fdotpre = np.zeros(len(tpre))
dfdotpre = np.zeros(len(tpre))

for k in range(len(tpre)):
    freqpre[k] = fpre0 + fdotpre0*((tpre[k]-tpre0)*day) +\
              0.5*fddotpre0*((tpre[k]-tpre0)*day)**2 +\
              (1./6.)*f3pre0*((tpre[k]-tpre0)*day)**3
    dfreqpre[k] = np.sqrt(dfpre0**2 + (dfdotpre0**2)*((tpre[k]-tpre0)*day)**2 +\
                      (1./4.)*(dfddotpre0**2)*((tpre[k]-tpre0)*day)**4 +\
                      (1./36.)*(df3pre0**2)*((tpre[k]-tpre0)*day)**6)
    if(tpre[k]>=tg):
        freqpre[k] += df0g
        
for k2 in range(len(tpre)):
    fdotpre[k2] = fdotpre0 + fddotpre0*((tpre[k2]-tpre0)*day) +\
             (0.5)*f3pre0*((tpre[k2]-tpre0)*day)**2
    dfdotpre[k2] = np.sqrt((dfdotpre0**2) + (dfddotpre0**2)*\
                       ((tpre[k2]-tpre0)*day)**2 +\
                       (0.5)*(df3pre0**2)*((tpre[k2]-tpre0)*day)**4)
fdotpre/=1e-11
dfdotpre/=1e-11

#Late post-outburst solution
tpost = [59281.3271828304211227,59282.3630374629484977,\
         59284.3601243647257749,59318.2758915396273921,\
         59318.3938688999791473,59319.7639456435683522,\
         59322.9839024242246772,59337.8332667654222261,\
         59373.8634878338586442,59409.2090374822590945,\
         59434.9130374140381976,59435.1491100361990538]
tpost = np.array(tpost)

tpost0 = 59312.30456614095
fpost0 = 3.0364659434503791222
dfpost0 = 2.8194525580344075695e-09
fdotpost0 = -6.6965735540487388964e-11
dfdotpost0 = 2.0550186635600577323e-15
fddotpost0 = 6.3843748007834583417e-21
dfddotpost0 = 5.2933602963824003213e-22

pepoch_em,_,_,f0em,df0em,f1em,df1em = np.loadtxt("segments_F0_F1.txt",\
                                                 skiprows=1,unpack=True)
fb_em = f0 + fdot0*((pepoch_em-T0)*day) + 0.5*fddot0*((pepoch_em-T0)*day)**2        
f0em = f0em - fb_em
fstar = fstar - fbase
f0em = -f0em
fstar = -fstar

freqpost = fpost0 + fdotpost0*((tpost-tpost0)*day) +\
            0.5*fddotpost0*((tpost-tpost0)*day)**2
dfreqpost = np.sqrt(dfpost0**2 + (dfdotpost0**2)*((tpost-tpost0)*day)**2 +\
                    (0.5*(dfddotpost0)*((tpost-tpost0)*day)**2)**2)
fdotpost = fdotpost0 + fddotpost0*(tpost-tpost0)*day 
dfdotpost = np.sqrt(dfdotpost0**2 + dfddotpost0**2*((tpost-tpost0)*day)**2)
fdotpost/=1e-11
dfdotpost/=1e-11

#Split residuals of different instruments
toa_nicer,_,_,_ = np.loadtxt("phase_resid_nicer.dat",skiprows=0,unpack=True)
toa_nustar,_,_,_ = np.loadtxt("phase_resid_nustar.dat",skiprows=0,unpack=True)
toa_xmm,_,_,_ = np.loadtxt("phase_resid_xmm.dat",skiprows=0,unpack=True)
toa_swift,_,_,_ = np.loadtxt("phase_resid_swift.dat",skiprows=0,unpack=True)

resid_nicer,resid_nustar,resid_xmm,resid_swift = [[],[],[],[]]
dresid_nicer,dresid_nustar,dresid_xmm,dresid_swift = [[],[],[],[]]
fnicer,fnustar,fxmm,fswift = [[],[],[],[]]
fbasenicer,fbasenustar,fbasexmm,fbaseswift = [[],[],[],[]]
fdotnicer,fdotnustar,fdotxmm,fdotswift = [[],[],[],[]]
dfnicer,dfnustar,dfxmm,dfswift = [[],[],[],[]]
dfdotnicer,dfdotnustar,dfdotxmm,dfdotswift = [[],[],[],[]]
newtoa_nicer,newtoa_nustar,newtoa_xmm,newtoa_swift = [[],[],[],[]]

for k in range(len(times)):
    
    for num1 in range(len(toa_nicer)):
        if(toa_nicer[num1]==times[k]):
            
            newtoa_nicer.append(times[k])
            rni = presid[k]-ystar[k]
            drni = (np.sqrt(dpresid[k]**2 + sigystar[k]**2))
            fbni = (fstar[k] - fbase[k])
            resid_nicer.append(rni)
            dresid_nicer.append(drni)
            fnicer.append(fstar[k])
            fbasenicer.append(fbni)
            fdotnicer.append(fdotstar[k]/1e-11)
            dfnicer.append(vfstar[k])
            dfdotnicer.append(vfdotstar[k]/1e-11)
    
    for num2 in range(len(toa_nustar)):
        if(toa_nustar[num2]==times[k]):
            
            newtoa_nustar.append(times[k])
            rnu = presid[k]-ystar[k]
            drnu = (np.sqrt(dpresid[k]**2 + sigystar[k]**2))
            fbnu = (fstar[k] - fbase[k])
            resid_nustar.append(rnu)
            dresid_nustar.append(drnu)
            fnustar.append(fstar[k])
            fbasenustar.append(fbnu)
            fdotnustar.append(fdotstar[k]/1e-11)
            dfnustar.append(vfstar[k])
            dfdotnustar.append(vfdotstar[k]/1e-11)

    for num3 in range(len(toa_swift)):
        if(toa_swift[num3]==times[k]):
            
            newtoa_swift.append(times[k])
            rs = presid[k]-ystar[k]
            drs = (np.sqrt(dpresid[k]**2 + sigystar[k]**2))
            fbs = (fstar[k] - fbase[k])
            resid_swift.append(rs)
            dresid_swift.append(drs)
            fswift.append(fstar[k])
            fbaseswift.append(fbs)
            fdotswift.append(fdotstar[k]/1e-11)
            dfswift.append(vfstar[k])
            dfdotswift.append(vfdotstar[k]/1e-11)

    for num4 in range(len(toa_xmm)):
        if(toa_xmm[num4]==times[k]):
            
            newtoa_xmm.append(times[k])
            rx = presid[k]-ystar[k]
            drx = (np.sqrt(dpresid[k]**2 + sigystar[k]**2))
            fbx = (fstar[k] - fbase[k])
            resid_xmm.append(rx)
            dresid_xmm.append(drx)
            fxmm.append(fstar[k])
            fbasexmm.append(fbx)
            fdotxmm.append(fdotstar[k]/1e-11)
            dfxmm.append(vfstar[k])
            dfdotxmm.append(vfdotstar[k]/1e-11)

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
pfluxxmm = [10**-11.8444,10**-11.7765,10**-11.7931]
dpfluxxmm = [1.9768e-13,1.54090e-13,1.112e-13]
pfluxxmm = np.array(pfluxxmm)
dpfluxxmm = np.array(dpfluxxmm)
pfluxxmm/=1e-12
dpfluxxmm/=1e-12

# timesswift,pfluxswift,dpfluxswift = np.loadtxt("cfluxes_swift.dat",\
#                                                 skiprows=0,unpack=True)
# pfluxswift/=1e-12
# dpfluxswift/=1e-12

# timesnustar,pfluxnustar,dpfluxnustar = np.loadtxt("cfluxes_nustar.dat",\
#                                                 skiprows=0,unpack=True)
# pfluxnustar/=1e-12
# dpfluxnustar/=1e-12

toutburst = 59055.0 + np.zeros(100)
ytoutburst = np.linspace(3.02,3.06,100)
ytoutburst2 = np.linspace(-7.8,-6.3,100)
ytoutburst3 = np.linspace(-10,100,100)

plt.figure()
plt.subplot(311)
plt.plot(toutburst,ytoutburst3,'k-')
plt.fill_betweenx(y=[-2,10],x1= [58000,58000], x2= [59055,59055],\
                  alpha=0.05,color="blue")
plt.fill_betweenx(y=[-8,100],x1= [59055,59055], x2= [59260,59260],\
                  alpha=0.05,color="darkorange")
plt.fill_betweenx(y=[-8,100],x1= [59260,59260], x2= [59550,59550],\
                  alpha=0.05,color="red")
plt.errorbar(ftoanicer,fluxnicer,yerr=dfluxnicer,fmt='b.',label="NICER")
plt.errorbar(timesxmm,pfluxxmm,yerr=dpfluxxmm,fmt='r.',label="XMM-Newton")
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
plt.ylim(0,10)
plt.ylabel(r"Flux [10$^{-12}$ erg cm$^{-2}$ s$^{-1}$]",fontsize=12)
plt.xlim(58300,59550)

newtoa_nicer = newtoa_nicer[0:-3]
fnicer = fnicer[0:-3]
dfnicer = dfnicer[0:-3]
fdotnicer = fdotnicer[0:-3]
dfdotnicer = dfdotnicer[0:-3]

t0nicer = 59063.74080453307
f0nicer = 3.037922666479065
fdot0nicer = -7.02754435278526e-11
fnicer = np.array(fnicer)
newtoa_nicer = np.array(newtoa_nicer)
fxmm = np.array(fxmm)
newtoa_xmm = np.array(newtoa_xmm)
fnustar = np.array(fnustar)
newtoa_nustar = np.array(newtoa_nustar)
fswift = np.array(fswift)
newtoa_swift = np.array(newtoa_swift)

fnicer += f0nicer + fdot0nicer*(newtoa_nicer-t0nicer)*day
fxmm += f0nicer + fdot0nicer*(newtoa_xmm-t0nicer)*day
fnustar += f0nicer + fdot0nicer*(newtoa_nustar-t0nicer)*day
fswift += f0nicer + fdot0nicer*(newtoa_swift-t0nicer)*day

plt.subplot(312)
plt.plot(toutburst,ytoutburst,'k-')
plt.errorbar(tpre,freqpre,yerr=dfreqpre,fmt='b.')
plt.errorbar(tpost,freqpost,yerr=dfreqpost,fmt='b.')
plt.errorbar(newtoa_nicer,fnicer,yerr=dfnicer,fmt='b.')
plt.errorbar(newtoa_xmm,fxmm,yerr=dfxmm,fmt='r.')
plt.errorbar(newtoa_nustar,fnustar,yerr=dfnustar,fmt='g.')
plt.errorbar(newtoa_swift,fswift,yerr=dfswift,fmt='k.')
plt.ticklabel_format(useOffset=False)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xticks([])
plt.fill_betweenx(y=[-8,6],x1= [58000,58000], x2= [59055,59055],\
                  alpha=0.05,color="blue")
plt.fill_betweenx(y=[-8,100],x1= [59055,59055], x2= [59260,59260],\
                  alpha=0.05,color="darkorange")
plt.fill_betweenx(y=[-8,100],x1= [59260,59260], x2= [59550,59550],\
                  alpha=0.05,color="red")
plt.ylabel(r"$\nu$ [Hz]",fontsize=12)
plt.xlim(58300,59550)
plt.ylim(3.034,3.044)

plt.subplot(313)
plt.fill_betweenx(y=[-8,6],x1= [58000,58000], x2= [59055,59055],\
                  alpha=0.05,color="blue")
plt.fill_betweenx(y=[-8,10],x1= [59055,59055], x2= [59260,59260],\
                  alpha=0.05,color="darkorange")
plt.fill_betweenx(y=[-8,100],x1= [59260,59260], x2= [59550,59550],\
                  alpha=0.05,color="red")
plt.plot(toutburst,ytoutburst2,'k-',label="Swift-BAT detection of ms burst")
plt.errorbar(newtoa_nicer,fdotnicer,yerr=dfdotnicer,fmt='b.',label="NICER")
plt.errorbar(newtoa_xmm,fdotxmm,yerr=dfdotxmm,fmt='r.',label="XMM-Newton")
plt.errorbar(newtoa_nustar,fdotnustar,yerr=dfdotnustar,fmt='g.',label="NuSTAR")
plt.errorbar(newtoa_swift,fdotswift,yerr=dfdotswift,fmt='k.',label="Swift")
plt.errorbar(tpre,fdotpre,yerr=dfdotpre,fmt='b.')
plt.errorbar(tpost,fdotpost,yerr=dfdotpost,fmt='b.')
plt.ylabel(r"$\dot{\nu}$ [10$^{-11}$ Hz/s]",fontsize=12)
plt.xlabel("Time [MJD]",fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlim(58300,59550)
plt.ylim(-7.4,-6.4)
# plt.legend(loc="best")
plt.subplots_adjust(hspace=0)
plt.show()

times = np.hstack((newtoa_nicer,newtoa_xmm,newtoa_nustar,newtoa_swift))
freqs = np.hstack((fnicer,fxmm,fnustar,fswift))
dfreqs = np.hstack((dfnicer,dfxmm,dfnustar,dfswift))
fdots = np.hstack((fdotnicer,fdotxmm,fdotnustar,fdotswift))
dfdots = np.hstack((dfdotnicer,dfdotxmm,dfdotnustar,dfdotswift))

pepoch_em = pepoch_em[8:-3]
f0em = f0em[8:-3]
df0em = df0em[8:-3]
f1em = f1em[8:-3]
df1em = df1em[8:-3]

pepoch_em = np.array(pepoch_em)
f0em = np.array(f0em)
f0em += f0nicer + fdot0nicer*(pepoch_em-t0nicer)*day

plt.figure()
plt.subplot(121)
plt.errorbar(pepoch_em,f0em,yerr=df0em,fmt='k.',\
             label="Classic segmented approach")
plt.errorbar(times,freqs,yerr=dfreqs,fmt='g.',label="GP regression")
plt.legend(loc="best")
plt.ylabel(r"$\nu$ [Hz]",fontsize=12)
plt.xlabel("Time [MJD]",fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.subplot(122)
plt.errorbar(pepoch_em,f1em/1e-11,yerr=df1em/1e-11,fmt='k.',\
              label="Classic segmented approach")
plt.errorbar(times,fdots,yerr=dfdots,fmt='g.',label="GP regression")
plt.xlabel("Time [MJD]",fontsize=12)
plt.ylabel(r"$\dot{\nu}$ [Hz s$^{-1}$]",fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(loc="best")
plt.show()

T = np.column_stack((tpre,freqpre,dfreqpre))
np.savetxt("timesol_pre_nicer.dat",T,fmt='%s',delimiter='   ')

V = np.column_stack((tpost,freqpost,dfreqpost))
np.savetxt("timesol_late_post_nicer.dat",V,fmt='%s',delimiter='   ')

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

tfdots = np.hstack((newtoa_nicer,newtoa_nustar,newtoa_xmm,newtoa_swift))
fdots = np.hstack((fdotnicer,fdotnustar,fdotxmm,fdotswift))
dfdots = np.hstack((dfdotnicer,dfdotnustar,dfdotxmm,dfdotswift))
tfdots,fdots,dfdots = zip(*sorted(zip(tfdots,fdots,dfdots)))

Z1 = np.column_stack((tfdots,fdots,dfdots))
np.savetxt("fdot_evol.dat",Z1,fmt='%s',delimiter='   ')
