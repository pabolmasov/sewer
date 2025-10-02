from numpy import *
import numpy.ma as ma

import matplotlib
from matplotlib.pyplot import *

from matplotlib import gridspec

import os
import sys
import glob

import h5py

# from scipy.optimize import root_scalar
# from scipy.integrate import simpson
# from scipy.integrate import cumulative_trapezoid as cumtrapz

from os.path import exists

from scipy.fft import fft, ifft, fftfreq, fft2, fftshift
# from scipy.signal import correlate

# HDF5 io:
import hio

# plotting
import plots

# simulating a wave moving to the right along z in pair relativistic plasma
# E, B, and v are allowed to have all the three components

# physical switches:
ifmatter = True # feedback 

ndigits = 2

# mesh:
nz = 2048
zlen = 20.
z = (arange(nz) / double(nz) - 0.5) * zlen
zhalf = (z[1:]+z[:-1])/2. # edges
dz = z[1] - z[0]
print("dz = ", dz)

# time
dt = dz * 0.1 # CFL in 1D should be not very small
tmax = 30.
dtout = 0.01
picture_alias = 30

# injection:
ExA = 0.0
EyA = 5.0
omega0 = 10.0
tpack = 1.0
tmid = tpack * 3. 

Bz = 0.

def Eleft(t):
    return sin(omega0 * t) * exp(-((t-tmid)/tpack)**2/2.)

def Bleft(t):
    return sin(omega0 * (t+dz/2.)) * exp(-((t+dz-tmid)/tpack)**2/2.)

def dBstep(Ex, Ey):
    dBx = zeros(nz) ;  dBy = zeros(nz)
    # dBx[0] = (Ey[0]-Eleft(t)*EyA) / dz
    # dBy[0] = -(Ex[0]-Eleft(t)*ExA) / dz
    # not updating the last cell!
    #using extended E arrays with the BCs
    
    dBx = (Ey[1:]-Ey[:-1]) / dz
    dBy = -(Ex[1:]-Ex[:-1]) / dz

    return dBx, dBy

def phiRL(uside, v):
    # uside has the size of nz+1
    # so does v
    
    allleft = (v[1:] >= 0.) * (v[:-1] >= 0.)
    allright = (v[1:] <= 0.) * (v[:-1] <= 0.)

    middle = 1-(allleft|allright)

    u = zeros(size(uside-1))

    if allleft.sum() > 0:
        u[allleft] = (uside[:-1])[allleft]

    if allright.sum() > 0:
        u[allright] = (uside[1:])[allright]

    if middle.sum() > 0:
        # slope limiter; chooses the smaller slope
        wleft = (abs(uside[:-1]) > abs(uside[1:])) * middle
        wright = (abs(uside[:-1]) <= abs(uside[1:])) * middle
        if wleft.sum() > 0:
            u[wleft] = (uside[1:])[wleft]
        if wright.sum() > 0:
            u[wright] = (uside[:-1])[wright]
        ## (abs(v[1:]) * uside[:-1] + abs(v[:-1]) * uside[1:]) / ( abs(v[1:]) + abs(v[:-1]))

    # print("left, middle, right = ", allleft.sum(), middle.sum(), allright.sum())
        
    return u
    
def dEstep(Bx, By, jx, jy, v):

    dEx = zeros(nz-1)
    dEy = zeros(nz-1)

    dEx = - (By[1:]-By[:-1])/dz + phiRL(jx, v) # (jx[1:]+jx[:-1])/2.
    dEy = (Bx[1:]-Bx[:-1])/dz + phiRL(jy, v) # (jy[1:]+jy[:-1])/2.
    
    return dEx, dEy

def dvstep(ux, uy, uz, n, Ex, Ey, Bx, By):
    # upwind integration

    gamma = sqrt(1. + ux**2 + uy**2 + uz**2)

    vx = ux / gamma ; vy = uy / gamma ; vz = uz / gamma

    vzhalf = (vz[1:]+vz[:-1])/2.
    
    dux_side = - vzhalf * (ux[1:]-ux[:-1])/dz # nz+1
    duy_side = - vzhalf * (uy[1:]-uy[:-1])/dz
    duz_side = - vzhalf * (uz[1:]-uz[:-1])/dz
    dn_side = -((n * vz)[1:] - (n*vz)[:-1])/dz
    
    dux = (vy * Bz - vz * By)[1:-1]
    duy = (vz * Bx - vx * Bz)[1:-1]
    duz = (vx * By - vy * Bx)[1:-1]
   
    dux += phiRL(Ex + dux_side, vzhalf)
    duy += phiRL(Ey + duy_side, vzhalf)
    duz += phiRL(duz_side, vzhalf)
    dn = phiRL(dn_side, vzhalf)
    
    return dux, duy, duz, dn 

def sewerrun():

    # HDF5 output:
    hout = hio.fewout_init('fout.hdf5',
                           {"ifmatter": ifmatter, "ExA": ExA, "EyA": EyA,
                            "omega0": omega0, "tpack": tpack, "tmid": tmid},
                           z, zhalf)
    
    # E on the edges, B in cell centres (Bz is not evolves, just postulated)
    Bx = zeros(nz)
    By = zeros(nz)
    
    Ex = zeros(nz-1)
    Ey = zeros(nz-1)
    Ex = zeros(nz-1)

    ux = zeros(nz)  ; uy = zeros(nz) ; uz = zeros(nz) ; n = ones(nz)
    
    t = 0. ; ctr = 0; figctr = 0
    
    while(t < tmax):

        Ex_ext = concatenate([[ExA * Eleft(t)], Ex, [By[-1]]])
        Ey_ext = concatenate([[EyA * Eleft(t)], Ey, [-Bx[-1]]])
        Bx_ext = concatenate([[EyA * Bleft(t)], Bx, [Bx[-1]]])
        By_ext = concatenate([[-ExA * Bleft(t)], By, [By[-1]]])
        ux_ext = concatenate([[ux[0]], ux, [ux[-1]]])
        uy_ext = concatenate([[uy[0]], uy, [uy[-1]]])
        uz_ext = concatenate([[uz[0]], uz, [uz[-1]]])
        n_ext = concatenate([[n[0]], n, [n[-1]]])
     
        if ifmatter:
            gamma = sqrt(1. + ux**2 + uy**2 + uz**2 )
            jx = n * ux/gamma ;     jy = n * uy/gamma ;         jz = 0.
        else:
            jx = 0. ; jy = 0. ; jz = 0.

        dBx, dBy = dBstep(Ex_ext, Ey_ext)
        dEx, dEy = dEstep(Bx, By, jx_ext, jy_ext, uz_ext)
        dux, duy, duz, dn = dvstep(ux_ext, uy_ext, uz_ext, n_ext, Ex_ext, Ey_ext, Bx_ext, By_ext)

        # preliminary time step
        Bx1 = Bx + dBx * dt/2. ; By1 = By + dBy * dt/2.
        Ex1 = Ex + dEx * dt/2. ; Ey1 = Ey + dEy * dt/2.
        ux1 = ux + dux * dt/2. ; uy1 = uy + duy * dt/2. ; uz1 = uz +  duz * dt/2. ; n1 = n + dn * dt/2.

        Ex_ext = concatenate([[ExA * Eleft(t+dt/2.)], Ex1, [By1[-1]]])
        Ey_ext = concatenate([[EyA * Eleft(t+dt/2.)], Ey1, [-Bx1[-1]]])
        Bx_ext = concatenate([[EyA * Bleft(t+dt/2.)], Bx1, [Bx1[-1]]])
        By_ext = concatenate([[-ExA * Bleft(t+dt/2.)], By1, [By1[-1]]])
        ux_ext = concatenate([[ux1[0]], ux1, [ux1[-1]]])
        uy_ext = concatenate([[uy1[0]], uy1, [uy1[-1]]])
        uz_ext = concatenate([[uz1[0]], uz1, [uz1[-1]]])
        n_ext = concatenate([[n1[0]], n1, [n1[-1]]])
        
        if ifmatter:
            gamma = sqrt(1. + ux1**2 + uy1**2 + uz1**2 )
            jx = n1 * ux1/gamma ;     jy = n1 * uy1/gamma ;         jz = 0.
        else:
            jx = 0. ; jy = 0. ; jz = 0.

        dBx, dBy = dBstep(Ex_ext, Ey_ext)
        dEx, dEy = dEstep(Bx1, By1, jx, jy, uz)
        dux, duy, duz, dn = dvstep(ux_ext, uy_ext, uz_ext, n_ext, Ex_ext, Ey_ext, Bx_ext, By_ext)

        # preliminary time step
        Bx = Bx + dBx * dt ; By = By + dBy * dt
        Ex = Ex + dEx * dt ; Ey = Ey + dEy * dt
        ux = ux + dux * dt ; uy = uy + duy * dt ; uz = uz +  duz * dt ; n = n + dn * dt

        t += dt ; ctr += 1
        if ctr%picture_alias==0:
            clf()
            plot(z, Bx, 'k-', label = r'$B_x$')
            plot(zhalf, Ey, 'r:', label = r'$E_y$')
            plot(zhalf[0]-dz, EyA * Eleft(t), 'bo', label = r'$E_y$ BC')
            xlabel(r'$z$') 
            title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
            legend()
            savefig('EB{:05d}.png'.format(figctr))
            clf()
            plot(z, uy, 'k-', label = r'$u^y$')
            plot(z, uz, 'r:', label = r'$u^z$')
            xlabel(r'$z$') 
            title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
            legend()
            savefig('uyz{:05d}.png'.format(figctr))
            clf()
            plot(uy, uy**2/2., 'r-')
            plot(uy, uz, 'k.')
            xlabel(r'$u^y$')   ;   ylabel(r'$u^z$') 
            title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
            savefig('GO{:05d}.png'.format(figctr))

            # HDF5 dump:
            hio.fewout_dump(hout, figctr, t, (Ex, Ey), (Bx, By), (ux, uy, uz), n)            
            # print(figctr)
            ctr = 0 ; figctr += 1

    hout.close()
