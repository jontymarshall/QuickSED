#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:37:09 2021

@author: jonty
"""

import numpy as np

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

class QuickSED:
    
    def __init__(self):
        #directory
        self.directory = ''
        self.prefix = 'new_model'
        self.obsv = False
        #wavelength grid
        self.wave_min = 1e-1
        self.wave_max = 1e4
        self.nwave = int(101)
        #stellar model
        self.dstar = 10.0
        self.tstar = 5770.0
        self.rstar = 1.0
        #disc model
        self.fdisc = [1e-4,1e-4]
        self.tdisc = [300.0,50.0]
        self.lam0  = [200.0,200.0]
        self.beta  = [1.5,1.5]
        #outputs
        self.sed_star = []
        self.sed_disc = []
        self.sed_total = 0.0
    
    def plam(wave,temp):
        """
        Planck function, returns B(lam,T)
        """
        
        a = 2.0*h*c**2
        b = h*c/(wave*k*temp)
        intensity = a/ ( (wave**5) * (np.exp(b) - 1.0) )
        
        return intensity
    
    def wave(self):
        """
        Create wavelength grid based on model parameters.

        """
        
        wavelengths = np.logspace(np.log10(self.wave_min), np.log10(self.wave_max), num=self.nwave,endpoint=True,base=10.0)
        
        self.sed_wave = wavelengths
        
    def star(self):
        """
        Create stellar blackbody model based on model parameters.
        """

        self.lstar = 4.*np.pi*(self.rstar*rsol)**2*sb*self.tstar**4 / lsol
        
        QuickSED.wave(self)
        photosphere = (self.rstar*rsol)**2*np.pi*QuickSED.plam(self.sed_wave*um,self.tstar)

        self.sed_star = photosphere
                
    def disc(self):
        """
        Create disc modified blackbody components according to the model inputs.

        """
        
        self.sed_disc = np.zeros((len(self.tdisc),len(self.sed_wave)))
        self.sed_total = np.zeros(len(self.sed_wave))
        for i in range(0, self.ndisc):
            modified = np.where(self.sed_wave >= self.lam0[i]) 
            emission = QuickSED.plam(self.sed_wave*um, self.tdisc[i])        
            emission[modified] = emission[modified]*(self.lam0[i]/self.sed_wave[modified])**self.beta[i]
            emission = emission*(self.fdisc[i]*np.trapz(self.sed_star)/np.trapz(emission))
            
            self.sed_disc[i,:] = emission
            self.sed_total += emission
        
        
    def plot(self):
        """
        Create a plot of the model SED (with observations if provided).

        """
        import matplotlib.pyplot as plt
                
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        
        sc = 1e3*1e26*(1./(self.dstar*pc)**2)*(self.sed_wave*um)**2 /c
        
        ax.loglog(self.sed_wave, self.sed_star*sc, color='grey',linestyle=':',label='Stellar model')
        for i in range(0, self.ndisc):
            ax.loglog(self.sed_wave, self.sed_disc[i]*sc, color='grey',linestyle='--',label='Mod. BB '+str(i+1))
        ax.loglog(self.sed_wave, (self.sed_star + self.sed_total)*sc, color='black',linestyle='-',label='Total')
        if self.obsv == True:
            ax.errorbar(self.owav,self.oflx,yerr=self.ounc,marker='o',color='black',linestyle='',label='Data')
        ax.set_xlabel(r'Wavelength ($\mu$m)')
        ax.set_ylabel(r'Flux density (mJy)')
        ax.set_xlim(self.wave_min,self.wave_max)
        if np.max(self.sed_star*sc) > np.max((self.sed_total*sc)):
            ax.set_ylim(10**(np.log10(np.max((self.sed_disc*sc))) - 6),10**(np.log10(np.max(self.sed_star*sc)) + 1))
        else:
            ax.set_ylim(10**(np.log10(np.max(self.sed_star*sc)) - 6),10**(np.log10(np.max((self.sed_disc*sc))) + 1))
        ax.legend()
        plt.tight_layout()
        fig.savefig(self.directory+self.prefix+'_QuickSED.png',dpi=200)
        #plt.show()
        plt.close(fig)
        
        self.sed_plot = fig
    
    def read(self,filename,units='mJy'):
        """
        Read observations from a file and add them to the model object for plotting and fitting.
        Wavelengths should be in microns, fluxes and uncertainties should be in mJy.
        
        Parameters
        ----------
        units : TYPE, optional
            Units of flux - either mJy or Jy. Jy will * 10^3 for conversion. The default is 'mJy'.

        """
        from astropy.io import ascii
        
        data = ascii.read(self.directory+filename,format='csv',guess=None,comment='#')
        
        if units =='mJy':
            self.obsv = True
            self.owav = data['wave'].data
            self.oflx = data['flux'].data
            self.ounc = data['uncs'].data
        if units =='Jy':
            self.obsv = True
            self.owav = data['wave'].data
            self.oflx = 1e3*data['flux'].data
            self.ounc = 1e3*data['uncs'].data
            
    def phot(self,snr=3,wav=20):
        """
        Generate synthetic photometry from model SED for comparison with observations.

        Parameters
        ----------
        snr : TYPE, optional
            S/N cut for fitted values. The default is 3.
        wav : TYPE, optional
            Wavelength cut (>) for fitted values. The default is 20.

        """
        sc = 1e3*1e26*(1./(self.dstar*pc)**2)*(self.sed_wave*um)**2 /c
        
        good = np.where((self.oflx/self.ounc > snr) & (self.owav > wav))
        
        self.good = good
        self.mflx = np.interp(self.owav,self.sed_wave,self.sed_total*sc)
        
    def auto(self):
        """
        Generate synthetic photometry from a model calculated by QuickSED.
        """
        sc = 1e3*1e26*(1./(self.dstar*pc)**2)*(self.sed_wave*um)**2 /c
        self.obsv = True
        self.owav = np.logspace(np.log10(0.4),np.log10(1300),num=20,endpoint=True)
        #print(self.owav)
        self.oflx = np.interp(self.owav,self.sed_wave,(self.sed_star+self.sed_total)*sc)
        #print(self.oflx)
        for i in range(len(self.oflx)):
            self.oflx[i] = self.oflx[i] + self.oflx[i]*0.1*np.random.randn()
        self.ounc = abs(self.oflx*0.2*np.random.randn())