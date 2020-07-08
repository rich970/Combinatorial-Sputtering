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
def rate_k(Px,Py,gamma, Phi_kF):
    """
    Calculate the deposition rate in atoms/[m2.s] for a magnetron at some 
    postion 'k'.

    Parameters
    ----------
    Px : float
        Horizontal coordinate along x-axis across the wafer for which the rate
        is calculated.
    Py : float
        Vertical coordinate along y-axis across the wafer for which the rate
        is calculated.
    gamma : float
        Angle which describes the magnetron position in terms of the azimuthal
        angle with respect the wafer. 90 deg -> magnetron is below the wafer,
        0 deg -> magnetron to the right of the wafer, 180 deg -> magnetron to the
        left of the wafer.
    Phi_kF : float
        depostion rate for magnetron k at the focal point on the wafer.

    Returns
    -------
    Phi_kP : float
        depostions rate for magnetron k at the Cartesian point on the wafer (Px,Py).

    """
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

def draw_circle(radius, divs, n_divs, ax, alpha = 0.6, color = 'w--'):
    """
    Draws a circle on the figure

    Parameters
    ----------
    radius : float
        Radius of the circle to be draw.
    divs : float
        Size of the divisions for which the wafer is divided.
    n_divs : int
        Number of division for which the wafer is divided.
    ax : axes handle
        Handle of the axes where the circle will be drawn

    Returns
    -------
    None.

    """
    cen = (n_divs-1)/2
    x = cen+(radius/divs)*np.cos(np.linspace(-np.pi,np.pi,100))
    y = cen+(radius/divs)*np.sin(np.linspace(-np.pi,np.pi,100))   
    ax.plot(x,y,color, linewidth = 2, alpha = alpha)
    return

def mask_array(array, radius, divs):
    """ 
    The comp array is a sqaure array of compositions. mask_comp() sets all compositions
    in the array which are outside the wafer radius to zero. 
    
    Parameters
    ----------
    comp : numpy array
        3D array of compositions of size (Px, Py, 3)
    radius : float 
        Radius of the wafer to be desposited on.
    divs : float
        Size of the divisions for which the wafer is divided.
    
    Returns 
    -------
    The masked comp array, with compositions outside wafer radius set to zero. 
    """
    if len(array.shape) == 3:
        [lx,ly,lz]=np.shape(array)    
        mask = np.ones([lx,ly,lz]) #mask array of ones
        nx = (lx-1)/2  # (nx, ny) centre indices for the mask array
        ny = (ly-1)/2
    else:
        [lx,ly]=np.shape(array)    
        mask = np.ones([lx,ly]) #mask array of ones
        nx = (lx-1)/2  # (nx, ny) centre indices for the mask array
        ny = (ly-1)/2
    
    #Loop over indices. If the (nx,ny) mask index is outside the wafer radius, set to zero.
    for index, val in np.ndenumerate(mask):
        if ((index[0]-nx)**2 + (index[1]-ny)**2) > (radius/divs)**2:
            mask[index] = 0 
             
    #multiply the composition matrix by the mask to remove compositions outside wafer diamter
    array = array*mask 
    return array 

def label_comp(comp,divs,ax):
    """
    Iteratively labels the compostions in the comp array on the figure

    Parameters
    ----------
    comp : numpy array
        3D array of compositions of size (Px, Py, 3).
    divs : float
        Size of the divisions for which the wafer is divided.
    ax : axes handle
        Handle of the axes where the composition text labels are added

    Returns
    -------
    None.

    """
    n=3 #n=3 labels in molar fraction, n=1 labels in atomic fraction
    if divs < 8:
        divs = 8  #divs is used to set fontsize. Ensure this can't be below 8
        
    for i in range(len(comp)):
        for j in range(len(comp)):
            ax.text(j, i, str(np.round(n*comp[i, j, 0],2)) + '\n'
                          + str(np.round(n*comp[i, j, 1],2)) + '\n'
                          + str(np.round(n*comp[i, j, 2],2)),
                          ha="center", va="center", color="w", fontsize = divs) 
    return 


# =============================================================================
# Main code
# =============================================================================
plt.close('all')
fig = plt.figure(figsize=(13,13))
ax = plt.axes()

GT = 180 #growth time in [s]
radius = 76.2 # wafer radius [mm]
divs = 8 #size of composition division that make up the wafer grid
#find edge of bounding box by rounding radius to the nearest 10. +4.99 to ensure always rounds up.

#Calculate atomic densities from mass densities
A_rho = np.mean([8.90,7.87,8.91])/(1.66054*np.mean([55.845,58.933,58.693]))
B_rho = 2.70/(1.66054*26.982)
C_rho = 7.21/(1.66054*54.983)

edge = np.round(radius+4.99,-1)  
n_divs = int(1+(2*edge)/divs) #number of divisions in the wafer grid

q = np.linspace(-edge,edge,n_divs)
#Calculate the atomic flux for magnetrons A, B, C over the full area
Px, Py = np.meshgrid(q,q)

#note: rate used in rate_k is in atoms/[m2.s] - not [AA/s]! We convert using the atomic density
A = rate_k(Px,Py,90,1*A_rho)  
B = rate_k(Px,Py,210,1*B_rho)
C = rate_k(Px,Py,330,1*C_rho)

#Calculate the compositions for each element
A_comp = A/(A+B+C)
B_comp = B/(A+B+C)
C_comp = C/(A+B+C)

#Estimate the thickness
thickness = GT*((A+B+C)/(A_comp*A_rho + B_comp*B_rho + C_comp*C_rho))

#collate the individual compositions as RGB channels in a new array for plotting
comp = np.zeros([n_divs,n_divs,3])
comp[..., 0] = A_comp
comp[..., 1] = B_comp
comp[..., 2] = C_comp

#Mask the edges of the array which are beyond the boundaries of a circular wafer    
comp = mask_array(comp, radius, divs)
thickness = mask_array(thickness, radius, divs)

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

#Create legend box
ax.text(1, n_divs-2, 'CoFeNi' + '\n'+ 'Al' + '\n' + 'Mn',
        ha="center", va="center", backgroundcolor = 'w', color="k",
        fontstyle = 'italic', fontsize = divs) 

#Draw a circle to represent the substrate
circ=draw_circle(radius, divs, n_divs, ax)

  
fig = plt.figure(figsize=(13,13))
ax2 = plt.axes()
#ax2.imshow(thickness, interpolation = 'none', origin = 'lower')
level=np.linspace(int(np.min(thickness[thickness>0])),int(np.max(thickness)),num=25)
CS=ax2.contour(thickness,level)
ax2.clabel(CS, inline=1, fontsize=10)
#plot the collated composition array as an RGB image
ax2.set_xticks(np.arange(len(q)))
ax2.set_yticks(np.arange(len(q)))
ax2.set_xticklabels(np.round(q,2),rotation=-45) #round label to 2.d.p and change orientation
ax2.set_yticklabels(np.round(q,2))
ax2.set_xlabel('x-distance from magnetron focus [mm]')
ax2.set_ylabel('y-distance from magnetron focus [mm]') 
ax2.set_title('Film thickness map [$\AA$]')

#make a grid using the minor ticks as reference
ax2.set_xticks(np.arange(len(q))-0.5, minor=True)
ax2.set_yticks(np.arange(len(q))-0.5, minor=True)
ax2.grid(which="minor", color="w", linestyle='-', linewidth=1)
circ=draw_circle(radius, divs, n_divs, ax2, color ='k--')
plt.show