#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:21:44 2020

Code to calculate variations in compositions across a wafer when using
combinatorial sputtering from three magnetron sources. 

The source are orientated at different angles (alpha, gamma) with respect to
the substrate. The deposition rate is therefore not uniform across the wafer, 
which can be used to obtain composition variations across a single sample. 

@author: richard
"""
# =============================================================================
# Import modules
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Functions
# =============================================================================
def Phi_k(Px,Py,gamma, Phi_kF):
    #Phi_kF = 1   #Rate at the focus for magnetron k. 
    st_d = 120#mm substrate - target distance 
    n = 2    #directionality constant. Higher n means more tightly focused flux beam
    
    alpha = np.deg2rad(20) #angle between r_kF and focus normal
    gamma = np.deg2rad(gamma) #magnetron position on clock face. 90 deg = below substrate.
    r_FP = np.array([Px,
                     Py,
                     0])

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

def draw_circle(radius, divs, n_divs, ax):
    cen = (n_divs-1)/2
    x = cen+(radius/divs)*np.cos(np.linspace(-np.pi,np.pi,100))
    y = cen+(radius/divs)*np.sin(np.linspace(-np.pi,np.pi,100))   
    ax.plot(x,y,'w--', linewidth = 2, alpha = 0.6)
    return

def mask_comp(comp, radius, divs):
    [lx,ly,lz]=np.shape(comp)
    mask = np.ones([lx,ly,lz]) #mask array of ones
    nx = (lx-1)/2  # (nx, ny) centre indices for the mask array
    ny = (ly-1)/2
    
    #Loop over indices. If the (nx,ny) mask index is outside the wafer radius, set to zero.
    for index, val in np.ndenumerate(mask):
        if ((index[0]-nx)**2 + (index[1]-ny)**2) > (radius/divs)**2:
            mask[index] = 0 
             
    #multiply the composition matrix by the mask to remove compositions outside wafer diamter
    comp = comp*mask 
    return comp  

def label_comp(comp,divs,ax):
    for i in range(len(comp)):
        for j in range(len(comp)):
            text = ax.text(j, i, str(np.round(3* comp[i, j, 0],2)) + '\n'
                                  + str(np.round(3*comp[i, j, 1],2)) + '\n'
                                  + str(np.round(3*comp[i, j, 2],2)),
                                  ha="center", va="center", 
                                  color="w", fontsize = divs) 
    return 

# =============================================================================
# Main code
# =============================================================================
plt.close('all')
fig = plt.figure(figsize=(13,13))
ax = plt.axes()

radius = 76.2 # wafer radius [mm]
divs = 8 #size of composition division that make up the wafer grid
#find edge of bounding box by rounding radius to the nearest 10. +4.99 to ensure always rounds up.
edge = np.round(radius+4.99,-1)  
n_divs = int(1+(2*edge)/divs) #number of divisions in the wafer grid

q = np.linspace(-edge,edge,n_divs)
#Calculate the atomic flux for magnetrons A, B, C over the full area
Px, Py = np.meshgrid(q,q)
A = Phi_k(Px,Py,90,1.0)
B = Phi_k(Px,Py,210,0.58)
C = Phi_k(Px,Py,330,0.58)

#Calculate the compositions for each element
A_comp = A/(A+B+C)
B_comp = B/(A+B+C)
C_comp = C/(A+B+C)

#collate the individual compositions as RGB channels in a new array for plotting
comp = np.zeros([n_divs,n_divs,3])
comp[..., 0] = A_comp
comp[..., 1] = B_comp
comp[..., 2] = C_comp

#Mask the edges of the array which are beyond the boundaries of a circular wafer    
comp = mask_comp(comp, radius, divs)

#plot the collated composition array as an RGB image
ax.imshow(comp, interpolation = 'none', origin = 'lower')
ax.set_xticks(np.arange(len(q)))
ax.set_yticks(np.arange(len(q)))
ax.set_xticklabels(np.round(q,2),rotation=-45) #round label to 2.d.p and change orientation
ax.set_yticklabels(np.round(q,2))
ax.set_xlabel('x-distance from magnetron focus [mm]')
ax.set_ylabel('y-distance from magnetron focus [mm]') 
ax.set_title('Wafer composition map')

#make a grid using the minor ticks as reference
ax.set_xticks(np.arange(len(q))-0.5, minor=True)
ax.set_yticks(np.arange(len(q))-0.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=1)

# Loop over data dimensions and create text annotations.
label_comp(comp,divs,ax)

#Draw a circle to represent the substrate
circ=draw_circle(radius, divs, n_divs, ax)

  


