from numpy import *
import numpy.ma as ma

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
ifmatter = True
ifonedirection = True

# mesh:
nz = 4096
zlen = 20.
z = (arange(nz) / double(nz) - 0.5) * zlen
dz = z[1] - z[0]
print("dz = ", dz)

# time
# t = 0.
dt = dz * 0.5 # CFL in 1D should be not very small
tmax = 20.
dtout = 0.01
picture_alias = 10

# initial conditions (circularly polirized wave)
z0 = 5.0
f0 = 2.0
amp0 = 0.01
bbgdx = 0.0  ; bbgdy = 0.0 ; bbgdz = 0.1
bx0 = sin(2. * pi * f0 * z) * exp(-(z/z0)**6/2. * 0.) * amp0 + bbgdx
by0 =  - cos(2. * pi * f0 * z) * exp(-(z/z0)**6/2. * 0.) * amp0 * 0. + bbgdy
bz0 = z *  0. + bbgdz
bz = bz0

ax0 = cos(2. * pi * f0 * z) * exp(-(z/z0)**6/2. * 0.) * amp0 * 0.
ay0 =  sin(2. * pi * f0 * z) * exp(-(z/z0)**6/2. * 0.) * amp0
az0 = z * 0.
# 4-velocity
ux0 = 0. * z
uy0 = 0. * z
uz0 = 0. * z
n0 = ones(nz) * 1.0 # density ; let us keep it unity, meaning time is in omega_p units. Lengths are internally in c/f = 2pi c / omega units, that allows a simpler expression for d/dz 

def onestep(f, F_ax, F_ay, F_az, F_bx, F_by, F_ux, F_uy, F_uz, F_n, ifmatter):
    # one RK4 step
    # avoid interference with the globals!
    
    # essential grid quantities:
    ax = ifft(F_ax) ;    ay = ifft(F_ay) ;    az = ifft(F_az)
    bx = ifft(F_bx) ;    by = ifft(F_by) # ;    bz = ifft(F_bz)
    ux = ifft(F_ux) ;    uy = ifft(F_uy) ;    uz = ifft(F_uz)
    n = ifft(F_n) # n is n gamma
    gamma = sqrt(1.+ux**2+uy**2+uz**2)
    vx = ux/gamma ; vy = uy/gamma ; vz = uz/gamma
    
    # Maxwell equations:
    dF_bx = -1.j * f * F_ay # one Maxwell
    dF_by = 1.j * f * F_ax # one Maxwell
    dF_ax = 1.j * f * F_by  - fft(n * vx) * complex(ifmatter) # other Maxwell
    dF_ay = -1.j * f * F_bx - fft(n * vy) * complex(ifmatter) # other Maxwell
    dF_az = 0. * n # - fft(n * uz) * complex(ifmatter) # other Maxwell

    # hydrodynamics
    dF_ux = F_ax + fft( - vx * ifft(-1.j*f*F_ux) + vy * bz - vz * by)
    dF_uy = F_ay + fft( - vy * ifft(-1.j*f*F_uy) + vz * bx - vx * bz)
    dF_uz = F_az + fft( - vz * ifft(-1.j*f*F_uz) + vx * by - vy * bx)
    dF_n = 1.j * f * fft(vz*n) # - fft(n / gamma**2 * (ax * ux + ay * uy + az * uz ))

    return dF_ax, dF_ay, dF_az, dF_bx, dF_by, dF_ux, dF_uy, dF_uz, dF_n

def sewerrun():
    
    f = fftfreq(nz, d = dz / (2. * pi)) # Fourier mesh

    # Fourier images
    F_bx = fft(bx0)
    F_ax = fft(ax0)
    F_ux = fft(ux0)
    F_by = fft(by0)
    F_ay = fft(ay0)
    F_uy = fft(uy0)
    F_bz = fft(bz0)
    F_az = fft(az0)
    F_uz = fft(uz0)
    F_n  = fft(n0)

    if ifonedirection:
        # the square root is the correction for plasma dispersion
        F_ay[abs(f)>0.] = 1. * copy(F_bx)[abs(f)>0.] * sqrt(1.+(2.*pi*f)**(-2))[abs(f)>0.]
        F_ax[abs(f)>0.] = -1. * copy(F_by)[abs(f)>0.]  * sqrt(1.+(2.*pi*f)**(-2))[abs(f)>0.]
            
    t = 0.
    ctr = 0
    tstore = 0.
    tlist = []
    bxlist = []
    Fbxlist = []
    uzlist = []
    nlist = []
    
    # hyperdiffusion core
    fsq = (f * conj(f)).real
    hyperpower = 4.0
    hypercore = exp(-(fsq / (fsq.real).max() * (2.*pi)**2)**hyperpower * 2.) + 0.j

    # print(f)
    fout = open('sewerout.dat', 'w+')
    fout.write('# t -- z -- Bx \n')
    
    while t < tmax:
        if t > (tstore + dtout - dt):
            # save previous values
            F_bx_prev = F_bx
            ax_prev = ifft(F_ax) ;    ay_prev = ifft(F_ay)  ;    az_prev = ifft(F_az)
            bx_prev = ifft(F_bx) ;    by_prev = ifft(F_by)  # ;    bz = ifft(F_bz)
            ux_prev = ifft(F_ux) ;    uy_prev = ifft(F_uy)  ;    uz_prev = ifft(F_uz)
            n_prev = ifft(F_n)
            gamma_prev = sqrt(1.+ux_prev**2+uy_prev**2+uz_prev**2)
            
        # TODO: make it dictionaries or structures
    
        dF_ax1, dF_ay1, dF_az1, dF_bx1, dF_by1, dF_ux1, dF_uy1, dF_uz1, dF_n1 = onestep(f, F_ax, F_ay, F_az, F_bx, F_by, F_ux, F_uy, F_uz, F_n, ifmatter)
        dF_ax2, dF_ay2, dF_az2, dF_bx2, dF_by2, dF_ux2, dF_uy2, dF_uz2, dF_n2 = onestep(f, F_ax + dF_ax1/3. * dt, F_ay  + dF_ay1/3. * dt, dF_az1/3. * dt, F_bx + dF_bx1/3. * dt, F_by + dF_by1/3. * dt, F_ux + dF_ux1/3. * dt, F_uy + dF_uy1/3. * dt, F_uz  + dF_uz1/3. * dt, F_n  + dF_n1/3. * dt, ifmatter)
        dF_ax2, dF_ay2, dF_az2, dF_bx2, dF_by2, dF_ux2, dF_uy2, dF_uz2, dF_n2 = onestep(f, F_ax + dF_ax2 * 2./3. * dt, F_ay  + dF_ay2 * 2./3. * dt, dF_az2 * 2./3. * dt, F_bx + dF_bx2 * 2./3. * dt, F_by + dF_by2 * 2./3. * dt, F_ux + dF_ux2 * 2./3. * dt, F_uy + dF_uy2 * 2./3. * dt, F_uz  + dF_uz2 * 2./3. * dt, F_n  + dF_n2 * 2./3. * dt, ifmatter)    
   
        # time step:
        F_bx += (dF_bx1 * 0.25 + dF_bx2 * 0.75) * dt ;    F_by += (dF_by1 * 0.25 + dF_by2 * 0.75) * dt
        F_ax += (dF_ax1 * 0.25 + dF_ax2 * 0.75) * dt ;    F_ay += (dF_ay1 * 0.25 + dF_ay2 * 0.75) * dt ;    F_az += (dF_az1 * 0.25 + dF_az2 * 0.75) * dt
        F_ux += (dF_ux1 * 0.25 + dF_ux2 * 0.75) * dt ;    F_uy += (dF_uy1 * 0.25 + dF_uy2 * 0.75) * dt ;    F_uz += (dF_uz1 * 0.25 + dF_uz2 * 0.75) * dt
        F_n += (dF_n1 * 0.25 + dF_n2 * 0.75) * dt
        t += dt

        # dyperdiffusion:
        F_bx *= hypercore ;   F_by *= hypercore
        F_ax *= hypercore ;   F_ay *= hypercore  ;   F_az *= hypercore 
        F_ux *= hypercore ;   F_uy *= hypercore  ;   F_uz *= hypercore   ;    F_n *= hypercore
        
        if t > (tstore + dtout):
            print("t = ", t)

            if ctr%picture_alias==0:
                # Fourier spectrum:
                plots.fourier(f/2./pi, F_bx, f0, ctr)

            # print(F_bx.real.max(), F_bx.real.min())
            ax = ifft(F_ax) ;    ay = ifft(F_ay)  ;    az = ifft(F_az)
            bx = ifft(F_bx) ;    by = ifft(F_by)  # ;    bz = ifft(F_bz)
            ux = ifft(F_ux) ;    uy = ifft(F_uy)  ;    uz = ifft(F_uz)
            n = ifft(F_n)
            gamma = sqrt(1.+ux**2+uy**2+uz**2)
            # vx = ux/gamma ; vy = uy/gamma ; vz = uz/gamma
            
            print("Bx = ", bx.min(), '..', bx.max())
            print("Ey = ", ay.min(), '..', ay.max())

            # ASCII output
            for k in arange(size(bx)):
                fout.write(str(t) + ' ' + str(z[k]) + str(bx[k])+'\n')
            fout.flush()
            
            if ctr%picture_alias==0:
                plots.onthefly(z, (z+zlen/2.+t)%zlen-zlen/2., ax0, ay0, az0, bx0, by0, ax, ay, az, bx, by, ux, uy, uz, n/gamma, ctr, t)
                
            tlist.append(tstore)
            bxlist.append(bx_prev.real + ((tstore-(t-dt))/dt) * (bx-bx_prev).real)
            Fbxlist.append(F_bx_prev + ((tstore-(t-dt))/dt) * (F_bx-F_bx_prev))
            # print(len(Fbxlist))
            uzlist.append(uz_prev.real +  ((tstore-(t-dt))/dt) * (uz-uz_prev).real)
            nlist.append((n_prev/gamma_prev).real +  ((tstore-(t-dt))/dt) * (n/gamma - n_prev/gamma_prev).real)
            tstore += dtout
            ctr += 1

    fout.close()
            
    tlist = asarray(tlist)
    bxlist = asarray(bxlist)
    Fbxlist = asarray(Fbxlist, dtype = complex)
    uzlist = asarray(uzlist).real
    nlist = asarray(nlist).real

    nt = size(tlist)

    #    print(tlist[1]-tlist[0], dtout)
    print("dtout = ", (tlist[1:]-tlist[:-1]).min(), (tlist[1:]-tlist[:-1]).max())
    
    # ii = input('T')
    # print(Fbxlist.real.max(), Fbxlist.real.min())
    #
    # bxlist_FF = fft2(bxlist)

    # should we clean the time-averaged value
    #Fbxlist_mean = Fbxlist.mean(axis = 0)
    #for k in arange(nt):
    #    Fbxlist[k, :] -= Fbxlist_mean[k]
    
    bxlist_FF = fft(Fbxlist, axis = 0) 
    ofreq = fftfreq(size(tlist), dtout)

    print("omega = ", ofreq)
    print("k = ", f/2./pi)
    
    bxlist_FF = fftshift(bxlist_FF)
    ofreq = fftshift(ofreq)
    f = fftshift(f)
    
    # print("wavenumber shape = ", shape(f))
    # print("frequency shape = ", shape(ofreq))
    # print(shape(bxlist_FF))
    # ii = input('T')

    #    nthalf = nt//2
    #    nzhalf = nz//2
    
    print("omega = ", ofreq)
    print("k = ", f/2./pi)

    # ii = input('T')
    
    # saving the data
    hio.okplane_hout(ofreq, f/2./pi, bxlist_FF, hname = 'okplane_Bx.hdf', dataname = 'Bx')

    plots.show_nukeplane(f0 = f0)
    
    plots.maps(z, tlist, bxlist, uzlist, nlist, ctr)
    
# ffmpeg -f image2 -r 20 -pattern_type glob -i 'EB*.png' -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  -pix_fmt yuv420p -b 8192k EB.mp4
