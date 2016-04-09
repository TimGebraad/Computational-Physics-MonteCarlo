# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:30:59 2016

@author: Marienne Faro
"""
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import scipy
from scipy.optimize import curve_fit
import pickle
import math
import copy

#Load data of a particular file and unpack the dictionary data to the corresponding parameters
def LoadData (szFileName, bPlot=False):
    pkl_file = open(szFileName, 'rb')
    Data = pickle.load(pkl_file)
    pkl_file.close()

    Method  = Data['Method']
    Type    = Data['Type']
    E       = Data['Energy']
    Evar    = Data['Evar']
    R       = Data['Position']
    Param   = Data['Param']    
        
    print('E = ', E, '+/-', np.sqrt(Evar), 'at ', Param)
    return Method, Type, E, Evar, Param, R

def MorsePot(s, D, a,re, E0):
    return E0 + D*(np.exp(-2*a*(s-re))-2*np.exp(-a*(s-re)))
    
def MorsePotRoot (s, D, a, re, E0, E):
    return MorsePot(s, D, a, re, E0)-E

def HarmOsc (s, e0, k, s0,):
    return e0 + k*(s-s0)**2
    
def HarmOscRoot (s, e0, k, s0, E):
    return HarmOsc (s, e0, k, s0)-E
        
#Procedure for analyzing the ground state energy diagram for all data placed in a folder splitted into different method and fitted to a harmonic and Morse potential
def AnalyzeEnergy (szFolder):
    E    = {'VMC':[], 'DMC':[]}
    Evar = {'VMC':[], 'DMC':[]}
    s    = {'VMC':[], 'DMC':[]}
    beta = {'VMC':[], 'DMC':[]}
    color= {'VMC':'darkslateblue', 'DMC':'mediumvioletred'}
    
    #Load all the data files in the folder an put the data in the lists
    for subdir, dirs, files in os.walk(szFolder): #Loop through all files of the folder
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".out"):
                print('----------------------------------------------------------->')
                print ('Loaded file:', filepath)
                Method, Type, Ei, Evari, Param, R = LoadData(filepath)
                E[Method].append(Ei)
                Evar[Method].append(Evari)
                s[Method].append(Param[0])
                beta[Method].append(Param[3])
    
    #Convert list to numpy arrays
    for Method in s:
        E   [Method] = np.array(E   [Method])
        Evar[Method] = np.array(Evar[Method])
        s   [Method] = np.array(s   [Method])
        beta[Method] = np.array(beta[Method])
    
    #Set up the figure
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1]) 
    gs.update(hspace=0.0)
    axEn = plt.subplot(gs[0])
    axBeta = plt.subplot(gs[1])        
    axEn2 = inset_axes(axEn, width="60%", height="80%", loc=1)
    axEn2.set_position([0, 1, 0, 5])
    
    [smin, smax] = [1, 2]   #range to fit harmonic potential (and inset plot)
    [sminm,smaxm]= [0.9, 6] #range to fit Morse potential
    meff = 918              #reduced mass of protons
    
    for Method in color:
        axEn.errorbar(s[Method], E[Method], np.sqrt(Evar[Method]), color=color[Method], )
        
        #Select range of s values for which the harmonic fit needs to apply
        idx = np.intersect1d(np.where(smin<=s[Method]),np.where(smax>=s[Method]))
        S = s[Method][idx] 
        En= E[Method][idx]
        Envar = Evar[Method][idx]       
        
        axEn2.plot (S, En, color=color[Method], marker='.', ls='None', alpha=1)
        axEn2.errorbar (S, En, np.sqrt(Envar), color=color[Method], marker='.', ls='None', alpha=0.25)
        
        #Fit to harmonic oscillator
        popt, pcov = curve_fit (HarmOsc, S, En)
        perr = np.sqrt(np.diag(pcov))
        E0 = popt[0]
        uE0= perr[0]
        k  = popt[1]
        uk = perr[1]
        s0 = popt[2]
        us0= perr[2]
        
        print ('\n \n Fitting to harmonic oscillator for', Method, '\n E0:', E0, '+/-', uE0, '\n k:', k, '+/-', uk, '\n s0:', s0, '+/-', us0, '\n')
        axEn2.plot(S, HarmOsc(S, E0, k, s0), color=color[Method], ls=':')
        w = math.sqrt(popt[1]/meff)
        uw= w/k*uk/2
        print('Frequency', w, '+/-', uw)

        #Calculate and plot the vibrational modes
        for v in range(3):
            Ev = E0 + w*(0.5+v)
            uEv= uE0 + uw*w*(0.5+v)
            if v==0:
                Ev0=Ev
                print('E', v, '=', Ev, '+/-', uEv)
            print('E', v, '=', Ev-Ev0, '+/-', uEv)
            s1 = scipy.optimize.brentq(HarmOscRoot, smin-1, s0, (E0, k, s0, Ev))
            s2 = scipy.optimize.brentq(HarmOscRoot, s0, smax+1, (E0, k, s0, Ev))
            axEn2.plot([s1, s2], HarmOsc([s1, s2], E0, k, s0), color=color[Method], ls=':', alpha=0.5)
        
        #Select range of s values for which the Morse fit needs to apply
        idx = np.intersect1d(np.where(sminm<=s[Method]),np.where(smaxm>=s[Method]))
        S = s[Method][idx] 
        En= E[Method][idx]
        Envar = Evar[Method][idx]

        #Fit to Morse potential
        popt, pcov = curve_fit (MorsePot, S, En)
        perr = np.sqrt(np.diag(pcov))
        D = popt[0]
        uD= perr[0]
        a = popt[1]
        ua= perr[1]
        re= popt[2]
        ure=perr[2]
        E0 =popt[3]
        uE0=perr[3]        
        
        print ('\n \n Fitting to Morse potential for', Method, '\n D', D, '+/-', uD, '\n a', a, '+/-', ua, '\n re', re, '+/-', ure, '\n E0', E0, '+/-', uE0, '\n')
        axEn2.plot(S, MorsePot(S, D, a, re, E0), color=color[Method])        
        w = a*math.sqrt(2*D/meff)
        uw = w*math.sqrt((ua/a)**2+(0.5*uD/D)**2)
        print('Frequency', w, '+/-', uw)
        
        #Calculate and plot the vibrational modes
        v0 = w/(2*math.pi) 
        uv0= uw/(2*math.pi)
        Emin =  MorsePot(popt[2], popt[0], popt[1], popt[2], popt[3])
        uEmin = uD+uE0  #approximation
        print('Minimum', Emin, '+/-', uEmin)
        for v in range(3):
            Ev = Emin + 2*math.pi*v0*(v+0.5)-(2*math.pi*v0*(v+0.5))**2/(4*D)
            uEv = uEmin + 2*math.pi*v0*(v+0.5)*uv0 + (2*math.pi*v0*(v+0.5))**2/(4*D)*math.sqrt((2*uv0/v0)**2+(uD/D)**2)
            if v==0:
                Ev0 = Ev
                print('E', v, '=', Ev, '+/-', uEv)
            print('E', v, '=', Ev-Ev0, '+/-', uEv)
            s1 = scipy.optimize.brentq(MorsePotRoot, smin-1, re, (D, a, re, E0, Ev))
            s2 = scipy.optimize.brentq(MorsePotRoot, re, smax+1, (D, a, re, E0, Ev))
            axEn2.plot([s1, s2], MorsePot([s1, s2], D, a, re, E0), color=color[Method],  alpha=0.5)

    #Plot the corresponding beta
    axBeta.plot(s['VMC'], beta['VMC'], marker='*', color='gold', markeredgecolor='goldenrod')
                
    #Tweaking of the figures
    axEn.set_ylabel('Energy (Hartree)')
    axEn.set_ylim([-1.25, 2.25])
    axBeta.set_ylabel('Beta')
    axBeta.set_xlabel('Internuclear distance (Bohr)')
    axEn2.set_xlim([smin, smax])
    axEn2.set_ylim([-1.19, -1.094])
        
    plt.show()    
    return
    
    
#Procedure to analyze the wavefunctions of different data files in a folder 
def AnalyzeWaveFunction (szFolder):
    SimData    = []
    
    for subdir, dirs, files in os.walk(szFolder): #Loop through all files of the folder
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".out"):
                print('----------------------------------------------------------->')
                print ('Loaded file:', filepath)
                Method, Type, Ei, Evari, Param, r = LoadData(filepath)
                SimData.append({'Type': Type, 'Param': Param, 'Pos': r})
                
    #Set data for the Hydrogen atom in a seperate variable for comparison
    for data in SimData:
        if data['Type']=='HydrogenAtom':
            xHyd = data['Pos'][0][0,:]                
          
    #Set up the plot
    fig = plt.figure()
    ax1 = fig.add_subplot(211)    
    ax2 = fig.add_subplot(212)
    color = ['limegreen', 'hotpink', 'crimson', 'darkred', 'purple', 'deepskyblue']
    szLegend = ()
    
    #Loop over all data and obtain the wavefunction depending on both r and x
    for i, data in enumerate(SimData):
        if data['Type']=='HydrogenAtom':
            r = la.norm(data['Pos'][0], axis=0)
            x = data['Pos'][0][0,:]
            szLegend += ('Hydrogen Atom',)
            
        elif data['Type']=='HydrogenMolecule':
            s = data['Param'][0]
            r1l = copy.deepcopy(data['Pos'][0])
            r1r = copy.deepcopy(data['Pos'][0])
            r2l = copy.deepcopy(data['Pos'][1])
            r2r = copy.deepcopy(data['Pos'][1])
            
            r1l[0,:] += s/2
            r1r[0,:] -= s/2
            r2l[0,:] += s/2
            r2r[0,:] -= s/2
            
            #Search for the closest proton
            r1 = np.minimum(la.norm(r1l, axis=0), la.norm(r1r, axis=0))
            r2 = np.minimum(la.norm(r2l, axis=0), la.norm(r2r, axis=0))
            r = np.concatenate((r1, r2))
            
            #Shift one of the protons to the center
            x = np.concatenate((data['Pos'][0][0,:], data['Pos'][1][0,:]))   
            x += s/2
            
            szLegend += ('Hydrogen Molecule (s=' + str(s) + ')',)
            
        elif data['Type']=='HeliumAtom':
            s= 0 
            r = np.concatenate((la.norm(data['Pos'][0], axis=0), la.norm(data['Pos'][0], axis=0)))
            x= data['Pos'][0][0,:]
            szLegend += ('Helium Atom',)
            
        #Plot r dependence
        Y, X = np.histogram(r, bins=100, normed=True, range=[0, 5])
        ax1.fill(np.concatenate((X[0:-1], [0])), np.concatenate((Y,[0])), color=color[i], alpha=0.2)
        ax1.plot(X[0:-1], Y, c=color[i])
        
        #Plot x dependence
        Y, X = np.histogram(x, bins=100, normed=True, range=[-3, 9])        
        ax2.fill(np.concatenate((X[0:-1], [0])), np.concatenate((Y,[0])), color=color[i], alpha=0.2)
        ax2.plot(X[0:-1], Y, c=color[i])
  
        #Compare to two Hydrogen atoms if type is Hydrogen molecule
        if data['Type']=='HydrogenMolecule':
            x = np.concatenate((xHyd, xHyd+s))
            Y, X = np.histogram(x, bins=100, normed=True, range=[-3, 9])
            ax2.plot(X[0:-1], Y, c=color[i], ls=':', lw=4)
      
    # Tweaking of the figure
    ax1.legend(szLegend)
    ax1.set_xlabel('r (Bohr)')
    ax1.set_ylabel('Probability density')
    ax2.set_xlabel('x (Bohr)')
    ax2.set_ylabel('Probability density')

if __name__ == '__main__':
    AnalyzeEnergy('Output/Sequence 1/')
    AnalyzeWaveFunction('Output/WaveFunction Large/')