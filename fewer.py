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
ifmatter = False # feedback
ifuz = False

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
omega0 = 5.0
tpack = 1.0
tmid = tpack * 3. 

Bz = 0.0
Bxbgd = 0.1
Bybgd = 0.0

def Aleft(t):
    return -sin(omega0 * (t -tmid + dz/2.)) * exp(-((t + dz/2.-tmid)/tpack)**2/2.) / omega0

def Eleft(t):
    return (cos(omega0 * (t-tmid)) - (t-tmid)/(omega0*tpack**2) * sin(omega0 * t)) * exp(-((t-tmid)/tpack)**2/2.)

def Bleft(t):
    return Eleft(t + dz/2.)
# sin(omega0 * (t-tmid+dz/2.)) * exp(-((t+dz/2.-tmid)/tpack)**2/2.)

def phiRL(uside, v):
    # uside has the size of nz+1
    # so does v

    if (size(v) != size(uside)):
        print(size(v), size(uside))
        ii = input("phiRL: v and u do not match")
    
    allleft = (v[1:] >= 0.) * (v[:-1] >= 0.)
    allright = (v[1:] <= 0.) * (v[:-1] <= 0.)

    middle = 1-(allleft|allright)

    u = zeros(size(uside)-1)

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
        
    return u    

def dBstep(Ex, Ey):
    dBx = zeros(nz) ;  dBy = zeros(nz) # ; v = ones(nz+1)
    # dBx[0] = (Ey[0]-Eleft(t)*EyA) / dz
    # dBy[0] = -(Ex[0]-Eleft(t)*ExA) / dz
    # not updating the last cell!
    #using extended E arrays with the BCs
    
    dBx = (Ey[1:]-Ey[:-1]) / dz
    dBy = -(Ex[1:]-Ex[:-1]) / dz

    return dBx, dBy

def dEstep(Bx, By, jx, jy, v):

    dEx = zeros(nz-1)
    dEy = zeros(nz-1)

    #    print("jx = ", size(jx), "; v = ", size(v))

    dEx = - (By[1:]-By[:-1])/dz
    dEy = (Bx[1:]-Bx[:-1])/dz
    
    if ifmatter:
        dEx += phiRL(jx, v)
        dEy += phiRL(jy, v)
    
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
   
    dux += phiRL(dux_side + Ex, vzhalf) 
    duy += phiRL(duy_side + Ey, vzhalf) 
    duz += phiRL(duz_side, vzhalf)
    dn = phiRL(dn_side, vzhalf)
    
    return dux, duy, duz, dn 

def sewerrun():

    
    # E on the edges, B in cell centres (Bz is not evolves, just postulated)
    Bx = zeros(nz)
    By = zeros(nz)

    Bx += Bxbgd
    By += Bybgd
    
    Ex = zeros(nz-1)
    Ey = zeros(nz-1)
    #     Ex = zeros(nz)

    ux = zeros(nz)  ; uy = zeros(nz) ; uz = zeros(nz) ; n = ones(nz)
    
    t = 0. ; ctr = 0; figctr = 0
    
    while ((t < tmax) & (abs(Bx).max() < (EyA * 100.))):

        if ifuz:
            uz0 = (ExA**2 + EyA**2) * Aleft(t)**2/2.
            uy0 = -EyA * Aleft(t)
        else:
            uz0 = minimum(uz[0], 0.)
            uy0 = uy[0] - Bxbgd * dz
            ux0 = ux[0] + Bybgd * dz

        n0 = 1.
            
        Ex_ext = concatenate([[ExA * Eleft(t)], Ex, [(By[-1] - Bybgd + Ex[-1])/2.]])
        Ey_ext = concatenate([[EyA * Eleft(t)], Ey, [(-Bx[-1] + Bxbgd +Ey[-1])/2.]])
        Bx_ext = concatenate([[EyA * Bleft(t)+Bxbgd], Bx, [Bx[-1]]])
        By_ext = concatenate([[-ExA * Bleft(t)+Bybgd], By, [By[-1]]])
        ux_ext = concatenate([[ux0], ux, [ux[-1]]])
        uy_ext = concatenate([[uy0], uy, [uy[-1]]])
        uz_ext = concatenate([[uz0], uz, [uz[-1]]])
        n_ext = concatenate([[n0], n, [n[-1]]])
     
        gamma = sqrt(1. + ux**2 + uy**2 + uz**2 )
        if ifmatter:
            # gamma = sqrt(1. + ux**2 + uy**2 + uz**2 )
            jx = -n * ux/gamma ;     jy = -n * uy/gamma ;         jz = 0.
        else:
            jx = 0. ; jy = 0. ; jz = 0.

        dBx, dBy = dBstep(Ex_ext, Ey_ext)
        dEx, dEy = dEstep(Bx, By, jx, jy, uz/gamma)
        dux, duy, duz, dn = dvstep(ux_ext, uy_ext, uz_ext, n_ext, Ex_ext, Ey_ext, Bx_ext, By_ext)

        # preliminary time step
        Bx1 = Bx + dBx * dt/2. ; By1 = By + dBy * dt/2.
        Ex1 = Ex + dEx * dt/2. ; Ey1 = Ey + dEy * dt/2.
        ux1 = ux + dux * dt/2. ; uy1 = uy + duy * dt/2. ; uz1 = uz +  duz * dt/2. ; n1 = n + dn * dt/2.

        if ifuz:
            uz0 = (ExA**2 + EyA**2) * Aleft(t+dt/2.)**2/2.
            uy0 = -EyA * Aleft(t+dt/2.)
        else:
            # uz0 = uz1[0] # minimum(uz1[0], 0.)
            # uy0 = uy1[0] # * 0.
            #            if abs(Bxbgd) > 0.:
            uz0 = minimum(uz1[0], 0.)
            uy0 = uy1[0] - Bxbgd * dz
            ux0 = ux1[0] + Bybgd * dz
            
        Ex_ext = concatenate([[ExA * Eleft(t+dt/2.)], Ex1, [(By1[-1] - Bybgd + Ex1[-1])/2.]])
        Ey_ext = concatenate([[EyA * Eleft(t+dt/2.)], Ey1, [(-Bx1[-1] +  Bxbgd +Ey1[-1])/2.]])
        Bx_ext = concatenate([[EyA * Bleft(t+dt/2.)+Bxbgd], Bx1, [Bx1[-1]]])
        By_ext = concatenate([[-ExA * Bleft(t+dt/2.)+Bybgd], By1, [By1[-1]]])
        ux_ext = concatenate([[ux0], ux1, [ux1[-1]]])
        uy_ext = concatenate([[uy0], uy1, [uy1[-1]]])
        uz_ext = concatenate([[uz0], uz1, [uz1[-1]]])
        n_ext = concatenate([[n0], n1, [n1[-1]]])
        
        gamma = sqrt(1. + ux**2 + uy**2 + uz**2 )
        if ifmatter:
            # gamma = sqrt(1. + ux**2 + uy**2 + uz**2 )
            jx = -n * ux/gamma ;     jy = -n * uy/gamma ;         jz = 0.
        else:
            jx = 0. ; jy = 0. ; jz = 0.

        dBx, dBy = dBstep(Ex_ext, Ey_ext)
        dEx, dEy = dEstep(Bx1, By1, jx, jy, uz/gamma)
        dux, duy, duz, dn = dvstep(ux_ext, uy_ext, uz_ext, n_ext, Ex_ext, Ey_ext, Bx_ext, By_ext)

        # preliminary time step
        Bx = Bx + dBx * dt ; By = By + dBy * dt
        Ex = Ex + dEx * dt ; Ey = Ey + dEy * dt
        ux = ux + dux * dt ; uy = uy + duy * dt ; uz = uz +  duz * dt ; n = n + dn * dt

        t += dt ; ctr += 1
        if ctr%picture_alias==0:
            
            # print("|By| <= ", abs(By).max())
            # print("|Ex| <= ", abs(Ex).max())
            clf()
            fig = figure()
            if abs(Bxbgd) > 0.:
                plot(z, Bx*0. + Bxbgd, 'k:', label = r'$B_x^{\rm bgd}$')
            plot(z, Bx, 'k-', label = r'$B_x$')
            if abs(By).max() > 0.:
                plot(z, By, 'k:', label = r'$B_y$')
            plot(zhalf, Ey, 'r-', label = r'$E_y$')
            if abs(Ex).max() > 0.:
                plot(zhalf, Ex, 'r:', label = r'$E_x$')
            plot(zhalf[0]-dz, EyA * Eleft(t), 'bo', label = r'$E_y$ BC')
            xlabel(r'$z$') 
            title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
            legend()
            fig.set_size_inches(12.,5.)
            savefig('EB{:05d}.png'.format(figctr))
            clf()
            plot(z, uy, 'k-', label = r'$u^y$')
            if abs(ux).max() > 0.:
                plot(z, ux, 'g--', label = r'$u^x$')
            plot(z, uz, 'r:', label = r'$u^z$')
            xlabel(r'$z$') 
            title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
            legend()
            fig.set_size_inches(12.,5.)
            savefig('uyz{:05d}.png'.format(figctr))
            clf()
            plot(uy, uy**2/2., 'r-')
            scatter(uy, uz, c = z)
            cb = colorbar()
            cb.set_label(r'$z$')
            xlabel(r'$u^y$')   ;   ylabel(r'$u^z$') 
            title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
            savefig('GO{:05d}.png'.format(figctr))
            close()
            
            # HDF5 dump:
            if figctr == 0:
                hout = hio.fewout_init('fout.hdf5',
                                       {"ifmatter": ifmatter, "ExA": ExA, "EyA": EyA,
                                        "omega0": omega0, "tpack": tpack, "tmid": tmid, "Bz": Bz, "Bx": Bxbgd},
                                       z, zhalf = zhalf)

            hio.fewout_dump(hout, figctr, t, (Ex, Ey), (Bx, By), (ux, uy, uz), n)            
            # print(figctr)
            ctr = 0 ; figctr += 1

    hout.close()
