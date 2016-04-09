# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:00:04 2016

@author: timge
"""
import math
import numpy as np
from numpy import linalg as la
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

"""-------------------------------------- Monte Carlo Methods ----------------------------------------- """
class MonteCarlo (object):
    def __init__(self):
        return
        
        #Determine the optimal value for a certain parameter with a minimum path finder
    def OptimizeParameter (self, Param, nParam=0, Nmax = 20, Threshold=0.001, bPlot=False, bSave=True):
        E = np.zeros((2,Nmax))
        ParamOpt =np.zeros(Nmax+1)
        ParamOpt[0] = Param[nParam]
        n = -1
        
        #Loop untill the difference drops below a certain threshold
        while (abs(E[0,n]-E[0,n-1]) > Threshold or n<2) and n<(Nmax-1):
            n+=1
            print('Parameter set to', ParamOpt[n], 'at iteration', n)
            E[0,n], E[1,n],Eder, r= self.PerformVMC(Param, bSave=False)
            ParamOpt[n+1] = ParamOpt[n] - 0.7*Eder
            Param[nParam] = ParamOpt[n+1]
            
        #Plot the path of the minimum finder in the appropriate parameter space if wanted
        if bPlot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(ParamOpt[0:n], E[0,0:n], np.sqrt(E[1,0:n]))
            plt.show()
            
        print('Optimum found of', E[0,n], '+/-', np.sqrt(E[1,n]), 'at', ParamOpt[n], 'after', n, 'iterations')
        
        if bSave:
            Data = {'Method': 'VMC', 'Type': self.Type, 'Param': Param, 'Energy': E[0,n], 'Evar': E[1,n], 'Position': r}  
            strName = 'Output/VMC - ' + self.Type + ' - Param ' + str(Param) + 'optimum.out'
            output = open(strName, 'wb')
            pickle.dump(Data, output)
            output.close()
            
        return E[0,n], E[1,n], ParamOpt[n+1]
        
        
        #Perform one integration by VMC with N walkers, T iterations and starting collecting data from Ts
    def PerformVMC (self, Param, N=400, T=30000, Ts=4000, d=0.1, bPlot=False, bSave=True):
        # Initialize walker positions within a cube of size 2
        r = np.random.uniform(-2, 2, (self.Nelectrons, self.Dim, N))
        EL = np.zeros((T,))
        Ed = np.zeros((T,))
        Ew = np.zeros((T,))
        
        for t in range(T):            
            #Create the new positions of the walkers by moving the particles of every walker within a cube of size d
            rnew = r + np.random.uniform(-d,d, (self.Nelectrons,self.Dim,N))
            
            #Determine whether the new position is accepted
            p = (self.Phi(Param, rnew)/self.Phi(Param, r))**2
            psuc = (p > np.ones(N)) + (p > np.random.random(N))
            r = (1-psuc)*r + psuc * rnew

            #Keep track of the energy
            E = self.LocalEnergy(Param, r)
            EL[t] = np.mean(E)
            phi = self.Phi(Param,r)
            Ew[t] = np.sum(phi**2*E)/np.sum(phi**2)
            
            dPsi = self.LogPsiDerivative(Param, r)
            Ed[t] = 2*(np.mean(E*dPsi)-EL[t]*np.mean(dPsi))
            print('Energy is', np.mean(EL[t]), 'at iteration', t)
            
        #Determine the mean and variance of the energy of the data that is selected to collect 
        E    = np.mean(EL[min(T-1,Ts):-1])
        Evar = np.var (EL[min(T-1, Ts):-1])
        Eder = np.mean(Ed[min(T-1, Ts):-1])
        print ('Energy is', E,'(', np.mean(Ew[min(T-1, Ts):-1]), ') +/-', Evar, 'with derivative',Eder )        
        
        #Plot the distribution of the walkers if wanted
        if bPlot:
            fig = plt.figure()
            ax = fig.add_subplot(121, projection='3d')
            for k in range(self.Nelectrons):
                ax.scatter(r[k][0,:], r[k][1,:], r[k][2,:], c='k', s=2, alpha=0.5)
            if self.Type=='HydrogenMolecule':
                ax.scatter([-Param[0]/2, +Param[0]/2], [0, 0], [0, 0], c='r', s=100)
            else:
                ax.scatter(0, 0, 0, c='r', s=100)
            ax2 = fig.add_subplot(122)
            ax2.plot(EL)
            plt.show()
            
        #Save the relevant information if wanted
        if bSave:
            Data = {'Method': 'VMC', 'Type': self.Type, 'Param': Param, 'Energy': E, 'Evar': Evar, 'Position': r, 'Eweight': Ew[min(T-1, Ts):-1]}  
            strName = 'Output/VMC - ' + self.Type + ' - Param ' + str(Param) + '.out'
            output = open(strName, 'wb')
            pickle.dump(Data, output)
            output.close()
            
        return E, Evar, Eder, r
        
        
        #Perform an integration using the simple diffusion Monte Carlo method with an initial estimate, a target number of walkers
        # T timesteps and start collecting data after Ts.
    def PerformSimpleDMC (self, E0,  N=300, T=30000, Ts=4000, d=0.1, bPlot=False, bSave=True):        
        # Initialize walker positions within a cube of size 2
        r = np.random.uniform(-2, 2, (self.Nelectrons, self.Dim, N))
        ET = np.zeros((T+1,))
        tau = 0.05
        
        M=N
        ET[0] = E0
        for t in range(T):            
            #Create the new positions of the walkers by moving 1 of the particles of every walker within a cube of size d
            r += np.random.normal(0,1, (self.Nelectrons,self.Dim,M))*math.sqrt(tau)
            
            #Create and eliminate particles at their respective positions
            q = np.exp(-tau*(self.LocalPotential(r)-ET[t]))
            n  = np.trunc(q + np.random.random(len(q)))
            M = int(np.sum(n))
            rnew = np.zeros((self.Nelectrons, 3, M))
            k=0
            for i in range(len(n)):
                for j in range(int(n[i])):
                    rnew[:, :, k] = r[:, :, i]
                    k +=1

            r = copy.deepcopy(rnew)          
            
            #Update ET
            ET[t+1] = E0 + 0.5*math.log(N/M)
            print('Energy is', ET[t], 'at iteration', t, 'with', M, 'Walkers')

        #Determine the mean and variance of the energy of the data that is selected to collect 
        E    = np.mean(ET[min(T-1,Ts):-1])
        Evar = np.var (ET[min(T-1, Ts):-1])
        print ('Energy is', np.mean(ET[min(T-1, Ts):-1]), '+/-', np.var(ET[min(T-1, Ts):-1]), 'with', M, 'Walkers')        
        
        #Plot the distribution of the walkers if wanted
        if bPlot:
            fig = plt.figure()
            ax = fig.add_subplot(131, projection='3d')
            ax.scatter(r[1][0,:], r[1][1,:], r[1][2,:], c='b')
            ax2 = fig.add_subplot(132)
            ax2.plot(ET)
            ax3 = fig.add_subplot(133)
            ax3.hist(la.norm(r[0], axis=0))
            plt.show()
            
        #Save the relevant data if wanted
        if bSave:
            Data = {'Method': 'DMC', 'Type': self.Type, 'Energy': E, 'Evar': Evar, 'Position': r}  
            strName = 'Output/DMC - ' + self.Type + '.out'
            output = open(strName, 'wb')
            pickle.dump(Data, output)
            output.close()
            
        return E, Evar
         
         
        #Perform an integration using the diffusion Monte Carlo method with a guide funcion
        # input an initial estimate, a target number of walkers, T timesteps and start collecting data after Ts.
    def PerformGuidedDMC (self, Param, E0, N=6000, T=4000, Ts=400, bPlot=False, bSave=True):
        # Initialize walker positions within a cube of size 2
        r = np.random.uniform(-2, 2, (self.Nelectrons, self.Dim, N))
        ET = np.zeros((T+1,))
        tau = 0.01
        M = N
        
        for t in range(T):
            #Add the force and a random displacement with a certain transmission probability
            F = self.Force(Param, r)
            r += F*tau/2
            rnew = r + np.random.normal(0,1, (self.Nelectrons,self.Dim,M))*math.sqrt(tau)        
            Trans = np.minimum(1, (self.Transmission (rnew, r, F, tau)*self.Phi(Param,rnew)**2)/(self.Transmission (r, rnew, F, tau)*self.Phi(Param,r)**2))
            
            q = np.exp(-tau*((self.LocalEnergy(Param, r)+self.LocalEnergy(Param, rnew))/2-ET[t]))
            psuc = Trans > np.random.random((len(Trans),))
            r = (1-psuc)*r + psuc * rnew
            
            #Create and eliminate particles at their respective positions
            n = np.trunc(q + np.random.random(len(q)))
            M = int(np.sum(n))
            rnew = np.zeros((self.Nelectrons, self.Dim, M))
            k=0
            for i in range(len(n)):
                for j in range(int(n[i])):
                    rnew[:, :, k] = r[:, :, i]
                    k +=1

            r = copy.deepcopy(rnew)          
            
            #Update the estimation for the ground state energy
            ET[t+1] = E0 + 0.5*math.log(N/M)
            print('Energy is', ET[t], 'at iteration', t, 'with', M, 'Walkers')
            
        #Determine the mean and variance of the energy of the data that is selected to collect 
        E    = np.mean(ET[min(T-1,Ts):-1])
        Evar = np.var (ET[min(T-1, Ts):-1])
        print ('Energy is', np.mean(ET[min(T-1, Ts):-1]), '+/-', np.var(ET[min(T-1, Ts):-1]), 'with', M, 'Walkers')
        
        #Plot the distribution of the walkers if wanted
        if bPlot:
            fig = plt.figure()
            ax = fig.add_subplot(121, projection='3d')
            for k in range(self.Nelectrons):
                ax.scatter(r[k][0,:], r[k][1,:], r[k][2,:], c='darkred', linewidth=0, s=2, alpha=0.4)
            if self.Type=='HydrogenMolecule':
                s = Param[0]
                ax.scatter([-s/2, +s/2], [0, 0], [0, 0], c='yellow', linewidth=0, s=100)
            else:
                ax.scatter(0, 0, 0, c='yellow', linewidth=0, s=100)  
            
            ax.auto_scale_xyz([-3,3], [-3,3], [-3,3])
            ax.axis("off")
            ax2 = fig.add_subplot(222)
            ax2.plot(ET)
            ax3 = fig.add_subplot(224)
            ax3.hist(la.norm(r[0], axis=0))
            ax3.hist(la.norm(r[1], axis=0))
            plt.show()
        
        #Save the relevant data if wanted
        if bSave:
            Data = {'Method': 'DMC', 'Type': self.Type, 'Param': Param, 'Energy': E, 'Evar': Evar, 'Position': r}  
            strName = 'Output/DMC - ' + self.Type + ' - Param ' + str(Param) + '.out'
            output = open(strName, 'wb')
            pickle.dump(Data, output)
            output.close()
            
        return E, Evar
        
    def Transmission (self, rold, rnew, F, tau):
        dummy = np.sum(np.sum((rnew-rold-F*tau/2)**2, axis=0), axis=0)
        return 1/math.sqrt(2*math.pi*tau)*np.exp(-dummy/(2*tau))
        

        
"""-------------------------------------- Different Systems to Analyze --------------------------------- """      
class Harmonic (MonteCarlo):
    def __init__(self):
        self.Type = 'nD Harmonic'
        self.Dim = 1
        self.Nelectrons = 1
        return
        
    def LocalEnergy (self, alpha, r):
        if self.Dim==1:
            R = la.norm(r[0], axis=0)
            return alpha+R**2*(0.5-2*alpha**2)
        elif self.Dim==3:
            R = la.norm(r[0], axis=0)
            return 3*alpha+R**2*(0.5-2*alpha**2)
        return 0
    
    def Force (self, alpha, r):
        if self.Dim==1:
            return -4*alpha*r
        
    def LocalPotential(self, r):
        R = la.norm(r[0], axis=0)
        return 0.5*R**2
        
    def Phi (self, alpha, r):
        return np.exp(-alpha*la.norm(r[0], axis=0)**2)


class HydrogenAtom (MonteCarlo):
    def __init__(self):
        self.Type = 'HydrogenAtom'
        self.Dim = 3
        self.Nelectrons = 1
        return
        
    def LocalEnergy (self, alpha, r):
        R = la.norm(r[0], axis=0)
        return -1/R-0.5*alpha*(alpha-2/R)
        
    def LocalPotential(self, r):
        return -1/la.norm(r[0], axis=0)
        
    def Force(self, alpha, r):
        R = la.norm(r[0], axis=0)
        return -alpha*r/R
        
    def Phi (self, alpha, r):
        R = la.norm(r[0], axis=0)
        return np.exp(-alpha*R)
            
    def LogPsiDerivative (self, alpha, r):
        return -la.norm(r[0], axis=0)
        
        
class HeliumAtom (MonteCarlo):
    def __init__(self):
        self.Type = 'HeliumAtom'
        self.Dim = 3
        self.Nelectrons = 2
        return
    
    def LocalEnergy(self, alpha, r):
        RH1 = r[0]/la.norm(r[0], axis=0)
        RH2 = r[1]/la.norm(r[1], axis=0)
        R12 = la.norm(r[1]-r[0], axis=0)  
        
        EL  = -4
        EL += np.sum((RH1-RH2)*(r[0]-r[1]), axis=0)/(R12*(1+alpha*R12)**2)
        EL -= 1/(R12*(1+alpha*R12)**3)
        EL -= 1/(4*(1+alpha*R12)**4)
        EL += 1/R12
        return EL
        
    def Force (self, alpha, r):
        R12 = la.norm(r[0]-r[1], axis=0)
        F12 = alpha*(2+R12)/(1+alpha*R12)**2/R12*(r[0]-r[1])
        return -2*r+[F12, -F12]

    def Phi (self, alpha, r):
        R1 =  la.norm(r[0], axis=0)
        R2 =  la.norm(r[1], axis=0)
        R12 = la.norm(r[1]-r[0], axis=0)
        return np.exp(-2*R1)*np.exp(-2*R2)*np.exp(R12/(2*(1+alpha*R12)))
        
    def LogPsiDerivative (self, alpha, r):
        R12 = la.norm(r[1]-r[0], axis=0)
        return -R12**2/(2*(1+alpha*R12)**2)
        
        
class HydrogenMolecule (MonteCarlo):
    def __init__(self):
        self.Type = 'HydrogenMolecule'
        self.Dim = 3
        self.Nelectrons = 2
        return
        
        #Function to obtain the corresponding value for a for a certain s (get the inverse of s(a) = -a*log(1/a-1))
    def GetA (self, S):
        a = np.arange(0.01,0.9999999999,0.00001)
        s = -a*np.log(1/a-1)
        for i in range(len(a)-1):
            if S > s[i] and S < s[i+1]:
                A = (a[i]+a[i+1])/2
                print('Corresponding value of s=', S, 'is a=', A)
        return A
        
    def LocalEnergy (self, Param, r):
        [s, a, alpha, beta] = Param        
        
        R1L = la.norm(self.GetRelR(r[0], +s/2), axis=0)
        R1R = la.norm(self.GetRelR(r[0], -s/2), axis=0)
        R2L = la.norm(self.GetRelR(r[1], +s/2), axis=0)
        R2R = la.norm(self.GetRelR(r[1], -s/2), axis=0)
        R12 = la.norm(r[0]-r[1], axis=0)
        
        R1Lh = self.GetRelR(r[0], +s/2)/R1L
        R1Rh = self.GetRelR(r[0], -s/2)/R1R
        R2Lh = self.GetRelR(r[1], +s/2)/R2L
        R2Rh = self.GetRelR(r[1], -s/2)/R2R
        R12h = (r[0]-r[1])/R12
        
        phi1L = np.exp(-R1L/a)
        phi1R = np.exp(-R1R/a)
        phi2L = np.exp(-R2L/a)
        phi2R = np.exp(-R2R/a)
        
        E  = -1/a**2
        E += 1/(a*(phi1L+phi1R))*(phi1L/R1L + phi1R/R1R)
        E += 1/(a*(phi2L+phi2R))*(phi2L/R2L + phi2R/R2R)
        E -= 1/R1L + 1/R1R + 1/R2L + 1/R2R - 1/R12
        E += np.sum(((phi1L*R1Lh+phi1R*R1Rh)/(phi1L+phi1R)-(phi2L*R2Lh+phi2R*R2Rh)/(phi2L+phi2R))*R12h, axis=0)/(2*a*(1+beta*R12)**2)
        E -= ((4*beta+1)*R12+4)/(4*(1+beta*R12)**4*R12)
        
        E += 1/s #Proton energy
        return E
        
    def Force(self, Param, r):
        [s, a, alpha, beta] = Param  
        
        R1L = la.norm(self.GetRelR(r[0], +s/2), axis=0)
        R1R = la.norm(self.GetRelR(r[0], -s/2), axis=0)
        R2L = la.norm(self.GetRelR(r[1], +s/2), axis=0)
        R2R = la.norm(self.GetRelR(r[1], -s/2), axis=0)
        R12 = la.norm(r[0]-r[1], axis=0)
        
        R1Lh = self.GetRelR(r[0], +s/2)/R1L
        R1Rh = self.GetRelR(r[0], -s/2)/R1R
        R2Lh = self.GetRelR(r[1], +s/2)/R2L
        R2Rh = self.GetRelR(r[1], -s/2)/R2R
        R12h = (r[0]-r[1])/R12
        
        phi1L = np.exp(-R1L/a)
        phi1R = np.exp(-R1R/a)
        phi2L = np.exp(-R2L/a)
        phi2R = np.exp(-R2R/a)
        
        F12 = 2*R12h/(alpha*(1+beta*R12)**2)       
        F1  = -2/a*(phi1L*R1Lh+phi1R*R1Rh)/(phi1L+phi1R)
        F2  = -2/a*(phi2L*R2Lh+phi2R*R2Rh)/(phi2L+phi2R)        
        
        return np.array([F1+F12, F2-F12])
        
        #Function to obtain the relative postions to one the protons
    def GetRelR (self, r, dx):
        R = copy.deepcopy(r)
        R[0,:] += dx
        return R
        
    def Phi (self, Param, r):
        [s, a, alpha, beta] = Param
        R1L = la.norm(self.GetRelR(r[0], +s/2), axis=0)
        R1R = la.norm(self.GetRelR(r[0], -s/2), axis=0)
        R2L = la.norm(self.GetRelR(r[1], +s/2), axis=0)
        R2R = la.norm(self.GetRelR(r[1], -s/2), axis=0)
        R12 = la.norm(r[0]-r[1], axis=0)
        
        return (np.exp(-R1L/a)+np.exp(-R1R/a))*(np.exp(-R2L/a)+np.exp(-R2R/a))*np.exp(R12/(alpha*(1+beta*R12)))
    
    def LogPsiDerivative (self, Param, r):
        [s, a, alpha, beta] = Param  
        R12 = la.norm(r[0]-r[1], axis=0)
        return -alpha*R12**2/(alpha*(1+beta*R12))**2
                        
    def PerformSeries (self, s=np.arange(1, 2, 0.02), bPlot=True):
        EVMC = np.zeros((2,len(s)))
        EDMC = np.zeros((2,len(s)))
        
        for i in range(len(s)):
            print('s', s[i], '-----------------------------------------------------------------------')
            a = self.GetA(s[i])
            EVMC[0,i], EVMC[1,i], beta = self.OptimizeParameter([s[i], a, 2, 0.6], nParam=3, bPlot=False)
            EDMC[0,i], EDMC[1,i]       = self.PerformGuidedDMC ([s[i], a, 2, beta], EVMC[0,i])
        
        if bPlot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(s, EVMC[0,:], np.sqrt(EVMC[1,:]), c='g')
            ax.errorbar(s, EDMC[0,:], np.sqrt(EDMC[1,:]), c='r')
            plt.show()
        return
        
        
if __name__ == '__main__':
   MySystem = HydrogenMolecule()
   MySystem.PerformSeries()
