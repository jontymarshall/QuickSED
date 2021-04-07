#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:51:23 2021

@author: jonty
"""
from QuickSED import QuickSED

#constants
h = 6.626e-34
c = 299792458.0 # m/s
k = 1.38e-23
sb = 5.67e-8 #
au     = 1.495978707e11 # m 
pc     = 3.0857e16 # m
lsol   = 3.828e26 # W
rsol   = 6.96342e8 # m
MEarth = 5.97237e24 # kg

um = 1e-6 #for wavelengths in microns

model = QuickSED()
model.directory = '/Users/jonty/mydata/QuickSED/'
model.ndisc = 2
model.dstar = 38.85

## Generate model from file

#QuickSED.read(model,'photometry.txt',units='Jy')
QuickSED.wave(model)
QuickSED.star(model)
QuickSED.disc(model)
QuickSED.auto(model)


model.prefix = 'new_model'
QuickSED.plot(model)

## Run emcee
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

def lnprior(theta):
    
    if model.ndisc == 1:
        f1,t1,l1,b1 = theta
        
        if -9. < f1 < -3. and 1. < t1 < 3. and 0. < l1 < 3. and 0 < b1 < 2 :
            return 0.0
        
        return -np.inf
    
    elif model.ndisc == 2:
        f1,t1,f2,t2,l1,b1 = theta
        
        if -9. < f1 < -3. and 1. < t1 < 3. and -9. < f2 < -3. and 1. < t2 < 3. and 0. < l1 < 3. and 0 < b1 < 2 :
            return 0.0        
        
        return -np.inf

def lnlike(theta,model_wav,model_obs,model_unc):
    
    if model.ndisc == 1:
        f1,t1,l1,b1 = theta
        
        model.fdisc = [10**f1]
        model.tdisc = [10**t1]
        model.lam0  = [10**l1]
        model.beta  = [b1]

    elif model.ndisc == 2:
        f1,t1,f2,t2,l1,b1 = theta
        
        model.fdisc = [10**f1,10**f2]
        model.tdisc = [10**t1,10**t2]
        model.lam0  = [10**l1,10**l1]
        model.beta  = [b1,b1]
        
    QuickSED.disc(model)    
    QuickSED.phot(model,snr=3,wav=10)
    good = model.good
        
    #print(-0.5 * np.sum((model.mflx[good] - model_obs[good])**2/model_unc[good]**2) )
    #print(theta)
    return -0.5 * np.sum((model.mflx[good] - model_obs[good])**2/model_unc[good]**2)    

def lnprob(theta,model_wav,model_obs,model_unc):
    lp=lnprior(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,model_wav,model_obs,model_unc)

def run_emcee(sampler,pos,ndim,labels,steps=500,prefix=""):
    print("Running MCMC...")
    sampler.run_mcmc(pos,steps, rstate0=np.random.get_state())
    print("Done.")

    plt.clf()
    
    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 9))
    
    for i in range(ndim):
        axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
        axes[i].set_ylabel(labels[i])

    fig.tight_layout(h_pad=0.0)
    fig.savefig(prefix+"line-time.png")
    return sampler

def mcmc_results(sampler,ndim,percentiles=[16, 50, 84],burnin=200,labels="",prefix=""):

    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    print(samples.shape)
    fig = corner.corner(samples, color='blue',labels=labels[0:ndim],quantiles=[0.16, 0.5, 0.84],show_titles=True,cmap='blues')
    fig.savefig(prefix+"line-triangle.png")
    credible_interval=[]
    for i in range(ndim):
        credible_interval.append(np.percentile(samples[:,i], percentiles))
        credible_interval[i][2] -= credible_interval[i][1]
        credible_interval[i][0] = credible_interval[i][1] - credible_interval[i][0]
    
    print("MCMC results:")
    for i in range(ndim):
        print("{0}  = {1[1]} + {1[2]} - {1[0]}".format(labels[i],credible_interval[i]))
        
    #now produce output plot of the best fit SED
    if model.ndisc == 1:
        
        model.fdisc = [10**credible_interval[0][1]]
        model.tdisc = [10**credible_interval[1][1]]
        model.lam0  = [10**credible_interval[2][1]]
        model.beta  = [credible_interval[3][1]]

    elif model.ndisc == 2:
                
        model.fdisc = [10**credible_interval[0][1],10**credible_interval[2][1]]
        model.tdisc = [10**credible_interval[1][1],10**credible_interval[3][1]]
        model.lam0  = [10**credible_interval[4][1],10**credible_interval[4][1]]
        model.beta  = [credible_interval[5][1],credible_interval[5][1]]
    
    
    print(model.fdisc,model.tdisc,model.lam0,model.beta)
    
    QuickSED.wave(model)
    QuickSED.star(model)
    QuickSED.disc(model)      
    QuickSED.plot(model)


nwalkers = 100
nsteps = 100
nburn = int(0.8*nsteps)

if model.ndisc == 1 :
    ndim   = 4
    labels = [r'$f_{\rm disc}$',r'$T_{\rm disc}$',r'$\lambda_{0}$',r'$\beta$']
    pos    = [[ -4 + np.random.randn() ,2. + 0.2*np.random.randn(),
               2.3 + 0.5*np.random.randn() ,1.0 + 0.25*np.random.randn()]  for i in range(nwalkers)]

elif model.ndisc == 2 :
    ndim   = 6
    labels = [r'$f_{\rm disc,1}$',r'$T_{\rm disc,1}$',
              r'$f_{\rm disc,2}$',r'$T_{\rm disc,2}$',
              r'$\lambda_{0}$',r'$\beta$']
    pos    = [[ -4 + np.random.randn() ,1.3 + 0.2*np.random.randn(),
                -5 + np.random.randn() ,2.1 + 0.2*np.random.randn(),
               2.3 + 0.5*np.random.randn() ,1.0 + 0.5*np.random.randn()]  for i in range(nwalkers)]

model_wav = model.owav
model_obs = model.oflx
model_unc = model.ounc

print(model.fdisc,model.tdisc,model.lam0,model.beta)

model.prefix = 'new2_model'

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,args=(model_wav,model_obs,model_unc))
results = run_emcee(sampler,pos,ndim,labels,nsteps,prefix=model.prefix)

mcmc_results(results,ndim,labels=labels,burnin=nburn,prefix=model.prefix)
