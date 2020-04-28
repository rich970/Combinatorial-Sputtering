#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:21:44 2020

@author: richard
"""

import numpy as np
import matplotlib.pyplot as plt

def Phi_k(Px,Py,gamma, Phi_kF):
    #Phi_kF = 1   #Rate at the focus for magnetron k. 
    st_d = 120 #mm substrate - target distance 
    n = 2    #directionality constant. Higher n means more tightly focused flux beam
    
    alpha = np.deg2rad(20) #angle between r_kF and focus normal
    gamma = np.deg2rad(gamma)
    r_FP = np.array([Px,
                     Py,
                     0])
    r_FPmag = np.linalg.norm(r_FP)    

    r_kF = (st_d) * np.array([np.sin(alpha)*np.cos(gamma), 
                              np.sin(alpha)*np.sin(gamma),
                              np.cos(alpha)])
    r_kFmag = np.linalg.norm(r_kF)
    
    r_kP = r_kF + r_FP
    r_kPmag = np.linalg.norm(r_kP)

    #angle between r_Fk and r_Pk (in radians) using dot product definition
    cos_beta = np.dot(r_kF,r_kP)/(r_kFmag*r_kPmag)
    cos_theta = np.dot(r_kP,np.array([0,0,1]))/(r_kPmag)
    cos_alpha = np.dot(r_kF,np.array([0,0,1]))/(r_kFmag)
        
    Phi_kP = Phi_kF * (r_kFmag/r_kPmag)**2 * cos_theta*(cos_beta**n)/cos_alpha  
    return Phi_kP


plt.close('all')
fig, ax  = plt.subplots(1,2,figsize=(25,10))

x = np.linspace(-70,70,15)
y = np.linspace(-70,70,15)
Px, Py = np.meshgrid(x,y)
A = Phi_k(Px,Py,90,1)
B = Phi_k(Px,Py,210,0.5)
C = Phi_k(Px,Py,330,0.5)

A_comp = A/(A+B+C)
B_comp = B/(A+B+C)
C_comp = C/(A+B+C)

comp = np.zeros([15,15,3])
comp[..., 0] = A_comp
comp[..., 1] = B_comp
comp[..., 2] = C_comp

ax[0].imshow(comp, interpolation = 'none', origin = 'lower')
ax[0].set_xticks(np.arange(len(x)))
ax[0].set_yticks(np.arange(len(y)))
ax[0].set_xticklabels(x)
ax[0].set_yticklabels(y)

#make a grid using the minor ticks as reference
ax[0].set_xticks(np.arange(len(x))-0.5, minor=True)
ax[0].set_yticks(np.arange(len(y))-0.5, minor=True)
ax[0].grid(which="minor", color="w", linestyle='-', linewidth=1)

# Loop over data dimensions and create text annotations.
for i in range(len(y)):
    for j in range(len(x)):
        text = ax[0].text(j, i, str(np.round(3* comp[i, j, 0],1)) + '\n'
                              + str(np.round(3*comp[i, j, 1],1)) + '\n'
                              + str(np.round(3*comp[i, j, 2],1)),
                              ha="center", va="center", color="w")

ax[0].set_xlabel('x-distance from magnetron focus [mm]')
ax[0].set_ylabel('y-distance from magnetron focus [mm]') 
ax[0].set_title('Composition [%]')  

ax[1].plot(Px[1,:],100*A_comp[4,:], 'ro-')
ax[1].plot(Px[1,:],100*B_comp[4,:], 'go-')
ax[1].plot(Px[1,:],100*C_comp[4,:], 'bo-')
ax[1].set_xlabel('x-distance from magnetron focus [mm]')
ax[1].set_ylabel('Percentage composition [%]') 
plt.axis([-70,70,0,70])
plt.legend(['A','B','C'])

