from numpy import *
import numpy.ma as ma

from matplotlib import gridspec

# import os
# import sys
# import glob

import threading
import time

import h5py

# from scipy.optimize import root_scalar
# from scipy.integrate import simpson
# from scipy.integrate import cumulative_trapezoid as cumtrapz

from os.path import exists

import matplotlib
from matplotlib.pyplot import *
    
cmap = 'viridis'

from scipy.fft import fft, ifft, fftfreq, fft2, fftshift
from scipy.integrate import simpson, trapezoid
# from scipy.signal import correlate

from numpy.random import rand

# HDF5 io:
import hio

# plotting
import plots

# simulating a wave moving to the right along z in pair relativistic plasma
# E, B, and v are allowed to have all the three components

# physical switches:
ifmatter = False
ifonedirection = True
ifnoise = False
ifneutral = True
nvars = 14 # three magnetic fields, three electric fields, three velocities + density for two particle species.

# mesh:
nz = 4096
zlen = 2.*(2.*pi)
z = (arange(nz) / double(nz) - 0.5) * zlen
dz = z[1] - z[0]
print("dz = ", dz)
f = fftfreq(nz, d = dz / (2.*pi)) # Fourier mesh

# time
# t = 0.
dt = dz * 0.5 # CFL in 1D should be not very small
tmax = 5.
# dtout = 0.01
# dtout = ceil(dtout/dt) * dt
dtalias = 1 # int(ceil(dtout/dt)) # how often are we saving the results
dtout = dt * dtalias
print("dtout / dt = ", dtalias)
picture_alias = 10 # every n-th data point is plotted in time
plotalias = 5 # every n-th point in space (or wavenumber)

# initial conditions (circularly polirized wave)
z0 = 1.0
omega0 = 20. # frequency in omegap units
# f0 = omega0 
amp0 = 1.0
amplim = 100. * amp0 # if az and/or ay exceed this value (abs), the simulation is aborted
glim = amp0 * 100. # gamma limit
bbgdx = 0.0  ; bbgdy = 0.0 ; bbgdz = 0.0
bx0 = sin(omega0 * z) * exp(-(z/z0)**6/2. * 0. ) * amp0 + bbgdx
by0 =  - cos(omega0 * z) * exp(-(z/z0)**6/2. * 0. ) * amp0 * 0. + bbgdy
bz0 = z *  0. + bbgdz
bz = bz0

ax0 = cos(omega0 * z) * exp(-(z/z0)**6/2. * 0. ) * amp0 * 0.
ay0 =  sin(omega0 * z) * exp(-(z/z0)**6/2. * 0.) * amp0
az0 = copy(z) * 0.

if ifnoise:
    bx0 = rand(nz) * amp0 +bbgdx
    by0 = rand(nz) * amp0  * 0. +bbgdy
    bz0 = rand(nz) * amp0 +bbgdz
    ax0 = rand(nz) * amp0  * 0.
    ay0 = rand(nz) * amp0 
    az0 = rand(nz) * amp0 * 0.
    
# 4-velocity
ux0p = 0. * z
uy0p = 0. * z
uz0p = 0. * z
n0p = ones(nz) * 1.0
ux0e = 0. * z
uy0e = 0. * z
uz0e = 0. * z
n0e = ones(nz) * 1.0
# density ; let us keep it unity, meaning time is in omega_p units. Lengths are internally in  c / omega units, that allows a simpler expression for d/dz 

ntol = 0.1
    
# hyperdiffusion kernel TODO: move to global scope?
fsq = (f * conj(f)).real 
hyperpower = 2.0
cutofffactor = 0.5
hypershift = 0.0
hypercore = minimum(exp(hypershift-(fsq / (fsq.max() * cutofffactor))**hyperpower * dt), 1.0) + 0.j    

def energyEM(ax, ay, az, bx, by):
    '''
    energy check (grid-based)
    '''

    ee = (ax**2+ay**2+az**2+bx**2+by**2)/2.

    return trapezoid(ee, x = z)

def energyPA(n, ux, uy, uz):
    '''
    energy check (grid-based)
    '''
    ee = sqrt(ux**2 + uy**2 + uz**2+1.)

    return simpson(ee * n, x = z)

    
def fft_thread(inar, outar, ifinverse = False):
    if ifinverse:
        # outar = copy(inar)
        outar[:] = ifft(inar[:])
    else:
        # outar = copy(inar, dtype=complex)
        outar[:] = fft(inar[:])

def NL_thread(f, vz, Fui, addgrid, outar):
    # outar = zeros(nz, dtype=complex)
    outar[:] += fft( vz * ifft(1.j*f*Fui[:]) + addgrid)

def cont_thread(f, vn, dF):
    # dF = zeros(nz, dtype=complex)
    dF[:] = 1.j * f * fft(vn)
    
def multiplyby_thread(ar, kernel):
    ar[:] *= kernel[:]

def densitycleaning_thread(F_n, thentol):
    # ntol = abs(n).mean()
    n = zeros(nz)
    n[:] = ifft(F_n[:]).real
    w = (n<(thentol*abs(n).mean()))

    if w.sum() > 0:
        n[w] = ntol*abs(n[w]).mean()

        F_n[:] = fft(n[:])
    
def onestep(f, F_ax, F_ay, F_az, F_bx, F_by, F_uxp, F_uyp, F_uzp, F_np, F_uxe, F_uye, F_uze, F_ne, ifmatter):
    # one RK4 step
    # avoid interference with the globals!
    # global ax, ay, az, bx, by, bz, uxp, uyp, uzp, np, uxe, uye, uze, ne
    time_togrid_start = time.time()
    # parallel computation of grid quantities:
    
    ax = zeros(nz)  ;  ay = zeros(nz) ;  az = zeros(nz)
    bx = zeros(nz)  ;  by = zeros(nz) # ;  bz = zeros(nz)
    uxp = zeros(nz)  ;  uyp = zeros(nz) ;  uzp = zeros(nz) ;  np = zeros(nz) # make them global?
    uxe = zeros(nz)  ;  uye = zeros(nz) ;  uze = zeros(nz) ;  ne = zeros(nz)
    togrid_ax = threading.Thread(target = fft_thread, args = (F_ax, ax), kwargs = {'ifinverse': True})
    togrid_ay = threading.Thread(target = fft_thread, args = (F_ay, ay), kwargs = {'ifinverse': True})
    togrid_az = threading.Thread(target = fft_thread, args = (F_az, az), kwargs = {'ifinverse': True})
    togrid_bx = threading.Thread(target = fft_thread, args = (F_bx, bx), kwargs = {'ifinverse': True})
    togrid_by = threading.Thread(target = fft_thread, args = (F_by, by), kwargs = {'ifinverse': True})
    # togrid_bz = threading.Thread(target = fft_thread, args = (F_bz, bz), kwargs = {'ifinverse': True})
    togrid_uxp = threading.Thread(target = fft_thread, args = (F_uxp, uxp), kwargs = {'ifinverse': True})
    togrid_uyp = threading.Thread(target = fft_thread, args = (F_uyp, uyp), kwargs = {'ifinverse': True})
    togrid_uzp = threading.Thread(target = fft_thread, args = (F_uzp, uzp), kwargs = {'ifinverse': True})
    togrid_np = threading.Thread(target = fft_thread, args = (F_np, np), kwargs = {'ifinverse': True})
    togrid_uxe = threading.Thread(target = fft_thread, args = (F_uxe, uxe), kwargs = {'ifinverse': True})
    togrid_uye = threading.Thread(target = fft_thread, args = (F_uye, uye), kwargs = {'ifinverse': True})
    if not(ifneutral):
        togrid_uze = threading.Thread(target = fft_thread, args = (F_uze, uze), kwargs = {'ifinverse': True})
        togrid_ne = threading.Thread(target = fft_thread, args = (F_ne, ne), kwargs = {'ifinverse': True})
    else:
        ne[:] = np[:]
        uze[:] = uzp[:]
        
    togrid_ax.start() ;    togrid_ay.start()  ;    togrid_az.start() 
    togrid_bx.start() ;    togrid_by.start()  # ;    togrid_bz.start() 
    togrid_uxp.start() ;    togrid_uyp.start()  ;    togrid_uzp.start() ; togrid_np.start()
    togrid_uxe.start() ;    togrid_uye.start()
    if not(ifneutral):
        togrid_uze.start() ; togrid_ne.start()

    togrid_ax.join() ;    togrid_ay.join()  ;    togrid_az.join() 
    togrid_bx.join() ;    togrid_by.join()  # ;    togrid_bz.join() 
    togrid_uxp.join() ;    togrid_uyp.join()
    togrid_uzp.join() ; togrid_np.join()
    togrid_uxe.join() ;    togrid_uye.join()
    if not(ifneutral):
        togrid_uze.join() ; togrid_ne.join()
   
    time_togrid_end = time.time()
    togrid_time_diff = time_togrid_end - time_togrid_start # in seconds

    #    ax = ifft(F_ax) ;    ay = ifft(F_ay) ;    az = ifft(F_az)
    #    bx = ifft(F_bx) ;    by = ifft(F_by) # ;    bz = ifft(F_bz)
    #    uxp = ifft(F_uxp) ;    uyp = ifft(F_uyp) ;    uzp = ifft(F_uzp)
    #    np = ifft(F_np) # n is n gamma
    #    uxe = ifft(F_uxe) ;    uye = ifft(F_uye) ;    uze = ifft(F_uze)
    #    ne = ifft(F_ne) # n is n gamma
    
    gammap = sqrt(1.+uxp**2+uyp**2+uzp**2)
    vxp = uxp/gammap ; vyp = uyp/gammap ; vzp = uzp/gammap
    if not(ifneutral):
        gammae = sqrt(1.+uxe**2+uye**2+uze**2)
        vxe = uxe/gammae ; vye = uye/gammae ; vze = uze/gammae
    else:
        ne = np
        gammae = gammap ; vxe = -vxp ; vye = -vyp ; vze = vzp
        
    # currents:
    if ifmatter:
        jx = np * uxp - ne * uxe
        jy = np * uyp - ne * uye
        jz = np * uzp - ne * uze
    else:
        jx = z * 0.
        jy = z * 0.
        jz = z * 0.
        
    # Maxwell equations:
    dF_bx =  -1.j * f * copy(F_ay) 
    dF_by =  1.j * f * copy(F_ax)
    
    if ifmatter:
        dF_ax = 1.j * f * F_by  - fft(jx)  # other Maxwell
        dF_ay = -1.j * f * F_bx - fft(jy)  # other Maxwell
        if not(ifneutral):
            dF_az =   - fft(jz)
        else:
            dF_az = zeros(nz, dtype = complex)
    else:
        dF_ax =  1.j * f * copy(F_by) 
        dF_ay =  -1.j * f * copy(F_bx) 
        dF_az = zeros(nz, dtype = complex)
    #
    # NL_thread(f, vz, Fui, addgrid, outar)

    # electric force:
    dF_uxp = copy(F_ax) #+ fft( -vzp * ifft(-1.j*f*F_uxp) + vyp * bz - vzp * by)
    dF_uyp = copy(F_ay) #+ fft( -vzp * ifft(-1.j*f*F_uyp) + vzp * bx - vxp * bz)
    dF_uxe = -copy(F_ax) #+ fft( -vzp * ifft(-1.j*f*F_uxp) + vyp * bz - vzp * by)
    dF_uye = -copy(F_ay) #+ fft( -vzp * ifft(-1.j*f*F_uyp) + vzp * bx - vxp * bz)
    
    if ifmatter:
        dF_uzp = F_az[:] #+ fft( -vzp * ifft(-1.j*f*F_uzp) + vxp * by - vyp * bx)
        dF_uze = -F_az[:]
    else:
        dF_uzp = zeros(nz, dtype=complex)
        dF_uze = zeros(nz, dtype=complex)
    # dF_uze = -F_az #+ fft( -vzp * ifft(-1.j*f*F_uzp) + vxp * by - vyp * bx)
    
    time_NL_start = time.time()
    # adding non-linear terms (parallelized)
    NLxp_thread = threading.Thread(target = NL_thread, args = (f, vzp, F_uxp, vyp * bz - vzp * by, dF_uxp))
    NLyp_thread = threading.Thread(target = NL_thread, args = (f, vzp, F_uyp, vzp * bx - vxp * bz, dF_uyp))
    NLzp_thread = threading.Thread(target = NL_thread, args = (f, vzp, F_uzp, vxp * by - vyp * bx, dF_uzp))
    NLxe_thread = threading.Thread(target = NL_thread, args = (f, vze, F_uxe, -vye * bz + vze * by, dF_uxe))
    NLye_thread = threading.Thread(target = NL_thread, args = (f, vze, F_uye, -vze * bx + vxe * bz, dF_uye))
    NLze_thread = threading.Thread(target = NL_thread, args = (f, vze, F_uze, -vxe * by + vye * bx, dF_uze))
    
    dF_np = zeros(nz, dtype = complex) ; dF_ne = zeros(nz, dtype = complex)
    contp_thread = threading.Thread(target = cont_thread, args = (f, vzp * np, dF_np))
    conte_thread = threading.Thread(target = cont_thread, args = (f, vze * ne, dF_ne))

    NLxp_thread.start() ; NLyp_thread.start() ; NLzp_thread.start() ;     NLxe_thread.start() ; NLye_thread.start() ; NLze_thread.start()
    contp_thread.start() ; conte_thread.start()
    
    NLxp_thread.join() ; NLyp_thread.join() ; NLzp_thread.join() ;     NLxe_thread.join() ; NLye_thread.join() ; NLze_thread.join()
    contp_thread.join() ; conte_thread.join()
   
    time_NL_end = time.time()
    NL_time_diff = time_NL_end - time_NL_start
    
    #    dF_np = 1.j * f * fft(vzp*np) 
    #    dF_uxe = -F_ax + fft( -vze * ifft(-1.j*f*F_uxe) + vye * bz - vze * by)
    #    dF_uye = -F_ay + fft( -vze * ifft(-1.j*f*F_uye) + vze * bx - vxe * bz)
    #    dF_uze = -F_az + fft( -vze * ifft(-1.j*f*F_uze) + vxe * by - vye * bx)
    #    dF_ne = 1.j * f * fft(vze * ne) 

    # print("max |dF_ay|  = ", abs(dF_ay).max())
    # ii = input('F')
    
    return dF_ax, dF_ay, dF_az, dF_bx, dF_by, dF_uxp, dF_uyp, dF_uzp, dF_np, dF_uxe, dF_uye, dF_uze, dF_ne, (togrid_time_diff, NL_time_diff)

def sewerrun2():
    global f
    # global ax, ay, az, bx, by, bz, uxp, uyp, uzp, np, uxe, uye, uze, ne
    # global ax_prev, ay_prev, az_prev, bx_prev, by_prev, bz_prev, uxp_prev, uyp_prev, uzp_prev, np_prev, uxe_prev, uye_prev, uze_prev, ne_prev
    # performance control:
    togrid_time = 0.0 ;    NL_time = 0.0  ;  diff_time = 0.0 ; total_time = 0.0 ; plotting_time = 0.0 ; timestep_time = 0.0
    total_time_start = time.time()    
    
    # Fourier images (local within sewerrun2)
    F_bx = fft(bx0)
    F_ax = fft(ax0)
    F_uxp = fft(ux0p)
    F_uxe = fft(ux0e)
    F_by = fft(by0)
    F_ay = fft(copy(ay0))
    F_uyp = fft(uy0p)
    F_uye = fft(uy0e)
    F_bz = fft(bz0)
    F_az = fft(az0)
    F_uzp = fft(uz0p)
    F_uze = fft(uz0e)
    F_np  = fft(n0p)
    F_ne  = fft(n0e)

    F_bx_prev = copy(F_bx)
    
    # can we keep them global?
    ax = zeros(nz)  ;  ay = zeros(nz) ;  az = zeros(nz)
    bx = zeros(nz)  ;  by = zeros(nz) # ;  bz = zeros(nz)
    uxp = zeros(nz)  ;  uyp = zeros(nz) ;  uzp = zeros(nz) ;  np = zeros(nz) # make them global?
    uxe = zeros(nz)  ;  uye = zeros(nz) ;  uze = zeros(nz) ;  ne = zeros(nz)
    ax_prev = zeros(nz)  ;  ay_prev = zeros(nz) ;  az_prev = zeros(nz)
    bx_prev = zeros(nz)  ;  by_prev = zeros(nz) # ;  bz_prev = zeros(nz)
    uxp_prev = zeros(nz)  ;  uyp_prev = zeros(nz) ;  uzp_prev = zeros(nz) ;  np_prev = zeros(nz) # make them global?
    uxe_prev = zeros(nz)  ;  uye_prev = zeros(nz) ;  uze_prev = zeros(nz) ;  ne_prev = zeros(nz)

    ampmax = 0. # maximum field amplitude; used to control the perturbation growth and abort the calculation when it exceeds some fraction of the initial amplitudes
    gammamax = 0.
    
    if ifonedirection:
        # the square root is the correction for plasma dispersion
        if ifmatter:
            F_ay[abs(f)>0.] = 1. * copy(F_bx)[abs(f)>0.] * sqrt(1.+(f[abs(f)>0.]/2./pi)**(-2))
            F_ax[abs(f)>0.] = -1. * copy(F_by)[abs(f)>0.]  * sqrt(1.+(f[abs(f)>0.]/2./pi)**(-2))
            F_az[:] *= 0.
        else:
            F_ay[abs(f)>0.] = 1. * copy(F_bx)[abs(f)>0.]
            F_ax[abs(f)>0.] = -1. * copy(F_by)[abs(f)>0.]
        # F_uye[:] = -1.j * f / omega0**2 * F_ay[:] ;      F_uyp[:] = 1.j * f / omega0**2 * F_ay[:]
        F_uyp[:] = -fft(cos(omega0 * z) ) * amp0/omega0 ; F_uye[:] = fft(cos(omega0 * z) ) * amp0/omega0
        # F_uxe[:] = F_ax[:] ;      F_uxp[:] = -F_ax[:]
        F_uzp[:] = fft(ifft(F_uye) * ifft(F_uye) + ifft(F_uxe) * ifft(F_uxe))[:]/2.
        uyp = ifft(F_uyp) ; uye = ifft(F_uye)
        uxp = ifft(F_uxp) ; uxe = ifft(F_uxe)
        uzp = ifft(F_uzp) ; uze = ifft(F_uze)
        ax = ifft(F_ax) ; ay = ifft(F_ay)  ; az = ifft(F_az) 
        bx = ifft(F_bx) ; by = ifft(F_by)  ; bz = ifft(F_bz) 
        gammap = sqrt(1.+uxp**2+uyp**2+uzp**2)   ;     gammae = sqrt(1.+uxe**2+uye**2+uze**2)
        
    t = 0.
    ctr = 0 ; dtctr = 0
    # tstore = 0.
    tlist = []   ;   elist = []  ;  eplist = []
    bxlist = []
    Fbxlist = []
    uzplist = [] ; uzelist = []
    nplist = [] ;   nelist = []

    # print(abs(F_uyp).max(), (uyp.real).max())
    # ii = input('uyp')
    #for k in arange(nz):
    #    print(f[k], fsq[k] / fsq.max(), hypercore[k])
    plots.onthefly(z[::plotalias], (z[::plotalias]+zlen/2.)%zlen-zlen/2., ax0[::plotalias], ay0[::plotalias], az0[::plotalias],
                   bx0[::plotalias], by0[::plotalias], ax[::plotalias], ay[::plotalias], az[::plotalias], bx[::plotalias], by[::plotalias],
                   uxp[::plotalias], uyp[::plotalias], uzp[::plotalias], (np[::plotalias]/gammap[::plotalias],ne[::plotalias]/gammae[::plotalias]), -1, t, omega = omega0)  
    fout = open('sewerout.dat', 'w+')
    fout.write('# t -- z -- Bx \n')
    
    while (t < tmax) & (ampmax < amplim) & (gammamax < glim):
        toutflag = False
        dplotting_time = 0.
        if (dtctr % dtalias) == 0:
            toutflag = True
            dtctr = 0
            # save previous values (togrid parallelized)
            # F_bx_prev[:] = F_bx[:]
            # parallelized conversion to grid values:
            time_togrid_start = time.time()
            togrid_ax_prev = threading.Thread(target = fft_thread, args = (F_ax, ax_prev), kwargs = {'ifinverse': True})
            togrid_ay_prev = threading.Thread(target = fft_thread, args = (F_ay, ay_prev), kwargs = {'ifinverse': True})
            togrid_az_prev = threading.Thread(target = fft_thread, args = (F_az, az_prev), kwargs = {'ifinverse': True})
            togrid_bx_prev = threading.Thread(target = fft_thread, args = (F_bx, bx_prev), kwargs = {'ifinverse': True})
            togrid_by_prev = threading.Thread(target = fft_thread, args = (F_by, by_prev), kwargs = {'ifinverse': True})
            togrid_uxp_prev = threading.Thread(target = fft_thread, args = (F_uxp, uxp_prev), kwargs = {'ifinverse': True})
            togrid_uyp_prev = threading.Thread(target = fft_thread, args = (F_uyp, uyp_prev), kwargs = {'ifinverse': True})
            togrid_uzp_prev = threading.Thread(target = fft_thread, args = (F_uzp, uzp_prev), kwargs = {'ifinverse': True})
            togrid_np_prev = threading.Thread(target = fft_thread, args = (F_np, np_prev), kwargs = {'ifinverse': True})
            togrid_uxe_prev = threading.Thread(target = fft_thread, args = (F_uxe, uxe_prev), kwargs = {'ifinverse': True})
            togrid_uye_prev = threading.Thread(target = fft_thread, args = (F_uye, uye_prev), kwargs = {'ifinverse': True})
            if not(ifneutral):
                togrid_uze_prev = threading.Thread(target = fft_thread, args = (F_uze, uze_prev), kwargs = {'ifinverse': True})
                togrid_ne_prev = threading.Thread(target = fft_thread, args = (F_ne, ne_prev), kwargs = {'ifinverse': True})
            else:
                uze_prev[:] = uzp_prev[:]
                ne_prev[:] = np_prev[:]
                
            togrid_ax_prev.start() ;    togrid_ay_prev.start()  ;    togrid_az_prev.start() 
            togrid_bx_prev.start() ;    togrid_by_prev.start()  # ;    togrid_bz_prev.start() 
            togrid_uxp_prev.start() ;    togrid_uyp_prev.start()  ;    togrid_uzp_prev.start() ; togrid_np_prev.start()
            togrid_uxe_prev.start() ;    togrid_uye_prev.start()
            if not(ifneutral):
                togrid_uze_prev.start() ; togrid_ne_prev.start()
                
                
            togrid_ax_prev.join() ;    togrid_ay_prev.join()  ;    togrid_az_prev.join() 
            togrid_bx_prev.join() ;    togrid_by_prev.join()  # ;    togrid_bz.join() 
            togrid_uxp_prev.join() ;    togrid_uyp_prev.join()  ;    togrid_uzp_prev.join() ; togrid_np_prev.join()
            togrid_uxe_prev.join() ;    togrid_uye_prev.join()
            if not(ifneutral):
                togrid_uze_prev.join() ; togrid_ne_prev.join()
                
            gammap_prev = sqrt(1.+uxp_prev**2+uyp_prev**2+uzp_prev**2)
            if not(ifneutral):
                gammae_prev = sqrt(1.+uxe_prev**2+uye_prev**2+uze_prev**2)
            else:
                gammae_prev = gammap_prev
                
        # TODO: make it dictionaries or structures
    
        dF_ax1, dF_ay1, dF_az1, dF_bx1, dF_by1, dF_uxp1, dF_uyp1, dF_uzp1, dF_np1, dF_uxe1, dF_uye1, dF_uze1, dF_ne1, dtos1 = onestep(f, F_ax, F_ay, F_az, F_bx, F_by, F_uxp, F_uyp, F_uzp, F_np, F_uxe, F_uye, F_uze, F_ne, ifmatter)
        dF_ax2, dF_ay2, dF_az2, dF_bx2, dF_by2, dF_uxp2, dF_uyp2, dF_uzp2, dF_np2, dF_uxe2, dF_uye2, dF_uze2, dF_ne2, dtos2 = onestep(f, F_ax + dF_ax1/3. * dt, F_ay  + dF_ay1/3. * dt, F_az + dF_az1/3. * dt, F_bx + dF_bx1/3. * dt, F_by + dF_by1/3. * dt, F_uxp + dF_uxp1/3. * dt, F_uyp + dF_uyp1/3. * dt, F_uzp  + dF_uzp1/3. * dt, F_np  + dF_np1/3. * dt, F_uxe + dF_uxe1/3. * dt, F_uye + dF_uye1/3. * dt, F_uze  + dF_uze1/3. * dt, F_ne + dF_ne1/3. * dt, ifmatter)
        dF_ax3, dF_ay3, dF_az3, dF_bx3, dF_by3, dF_uxp3, dF_uyp3, dF_uzp3, dF_np3, dF_uxe3, dF_uye3, dF_uze3, dF_ne3, dtos3 = onestep(f, F_ax + dF_ax2 * 2./3. * dt, F_ay + dF_ay2 * 2./3. * dt, F_az + dF_az2 * 2./3. * dt, F_bx + dF_bx2 * 2./3. * dt, F_by + dF_by2 * 2./3. * dt, F_uxp + dF_uxp2 * 2./3. * dt, F_uyp + dF_uyp2 * 2./3. * dt, F_uzp  + dF_uzp2 * 2./3. * dt, F_np  + dF_np2 * 2./3. * dt, F_uxe + dF_uxe2 * 2./3. * dt, F_uye + dF_uye2 * 2./3. * dt, F_uze  + dF_uze2 * 2./3. * dt, F_ne  + dF_ne2 * 2./3. * dt, ifmatter)    
        
        # time step:
        timestep_start = time.time()
        F_bx += (dF_bx1 * 0.25 + dF_bx3 * 0.75) * dt ;    F_by += (dF_by1 * 0.25 + dF_by3 * 0.75) * dt
        F_ax += (dF_ax1 * 0.25 + dF_ax3 * 0.75) * dt ;
        # print(abs(F_ay).max())
        F_ay += (dF_ay1 * 0.25 + dF_ay3 * 0.75) * dt
        # print(abs(F_ay).max())
        # ii = input('ay')
        #  ;    F_az += (dF_az1 * 0.25 + dF_az2 * 0.75) * dt
        F_uxp += (dF_uxp1 * 0.25 + dF_uxp3 * 0.75) * dt ;    F_uyp += (dF_uyp1 * 0.25 + dF_uyp3 * 0.75) * dt ;    F_uzp += (dF_uzp1 * 0.25 + dF_uzp3 * 0.75) * dt
        F_np += (dF_np1 * 0.25 + dF_np3 * 0.75) * dt 
        F_uxe += (dF_uxe1 * 0.25 + dF_uxe3 * 0.75) * dt ;    F_uye += (dF_uye1 * 0.25 + dF_uye3 * 0.75) * dt

        if not(ifneutral):
            F_az += (dF_az1 * 0.25 + dF_az3 * 0.75) * dt
            F_uze += (dF_uze1 * 0.25 + dF_uze3 * 0.75) * dt
            F_ne += (dF_ne1 * 0.25 + dF_ne3 * 0.75) * dt
        else:
            F_ne[:] = F_np[:]
            F_uze[:] = F_uzp[:]
            F_az[:] *= 0.
            # print("dF = ", abs(dF_az1).max(), abs(dF_az2).max(), abs(F_az).max())
            
        t += dt ; dtctr += 1
        timestep_end = time.time()

        # what if density becomes negative? 
        thread_densitycleaning_np = threading.Thread(target = densitycleaning_thread, args = (F_np, ntol))
        thread_densitycleaning_ne = threading.Thread(target = densitycleaning_thread, args = (F_ne, ntol))

        thread_densitycleaning_np.start() ; thread_densitycleaning_ne.start()
        thread_densitycleaning_np.join() ; thread_densitycleaning_ne.join()

        # print("hypercore = ", hypercore.min(), hypercore.max())
        
        # hyperdiffusion:
        ddiff_time_start = time.time()
        thread_diff_ax = threading.Thread(target = multiplyby_thread, args = (F_ax, hypercore))
        thread_diff_ay = threading.Thread(target = multiplyby_thread, args = (F_ay, hypercore))
        thread_diff_az = threading.Thread(target = multiplyby_thread, args = (F_az, hypercore))
        thread_diff_bx = threading.Thread(target = multiplyby_thread, args = (F_bx, hypercore))
        thread_diff_by = threading.Thread(target = multiplyby_thread, args = (F_by, hypercore))
        thread_diff_uxp = threading.Thread(target = multiplyby_thread, args = (F_uxp, hypercore))
        thread_diff_uyp = threading.Thread(target = multiplyby_thread, args = (F_uyp, hypercore))
        thread_diff_uzp = threading.Thread(target = multiplyby_thread, args = (F_uzp, hypercore))
        thread_diff_np = threading.Thread(target = multiplyby_thread, args = (F_np, hypercore))
        thread_diff_uxe = threading.Thread(target = multiplyby_thread, args = (F_uxe, hypercore))
        thread_diff_uye = threading.Thread(target = multiplyby_thread, args = (F_uye, hypercore))
        thread_diff_uze = threading.Thread(target = multiplyby_thread, args = (F_uze, hypercore))
        thread_diff_ne = threading.Thread(target = multiplyby_thread, args = (F_ne, hypercore))

        thread_diff_ax.start() ; thread_diff_ay.start() ; thread_diff_az.start()
        thread_diff_bx.start() ; thread_diff_by.start() # ; thread_diff_bz.start()
        thread_diff_uxp.start() ; thread_diff_uyp.start() ; thread_diff_uzp.start()  ; thread_diff_np.start()
        thread_diff_uxe.start() ; thread_diff_uye.start() ; thread_diff_uze.start()  ; thread_diff_ne.start()

        thread_diff_ax.join() ; thread_diff_ay.join() ; thread_diff_az.join()
        thread_diff_bx.join() ; thread_diff_by.join() # ; thread_diff_bz.join()
        thread_diff_uxp.join() ; thread_diff_uyp.join() ; thread_diff_uzp.join()  ; thread_diff_np.join()
        thread_diff_uxe.join() ; thread_diff_uye.join() ; thread_diff_uze.join()  ; thread_diff_ne.join()
        # F_bx *= hypercore ;   F_by *= hypercore
        # F_ax *= hypercore ;   F_ay *= hypercore  ;   F_az *= hypercore 
        # F_uxp *= hypercore ;   F_uyp *= hypercore  ;   F_uzp *= hypercore   ;    F_np *= hypercore
        # F_uxe *= hypercore ;   F_uye *= hypercore  ;   F_uze *= hypercore   ;    F_ne *= hypercore
        ddiff_time_end = time.time()
        # print(abs(F_ay).max())
        # ii = input('ay')
       
        if toutflag:
            print("t = ", t)
            
            if ctr%picture_alias==0:
                # Fourier spectrum:
                plotting_time_start = time.time()
                plots.fourier(f[::plotalias], F_bx[::plotalias], omega0, ctr)
                plotting_time_end = time.time()
                dplotting_time = plotting_time_end - plotting_time_start
            else:
                dplotting_time = 0.
                
            # parallelized conversion to grid values:
            time_togrid_start = time.time()
            togrid_ax = threading.Thread(target = fft_thread, args = (F_ax, ax), kwargs = {'ifinverse': True})
            togrid_ay = threading.Thread(target = fft_thread, args = (F_ay, ay), kwargs = {'ifinverse': True})
            togrid_az = threading.Thread(target = fft_thread, args = (F_az, az), kwargs = {'ifinverse': True})
            togrid_bx = threading.Thread(target = fft_thread, args = (F_bx, bx), kwargs = {'ifinverse': True})
            togrid_by = threading.Thread(target = fft_thread, args = (F_by, by), kwargs = {'ifinverse': True})
            togrid_uxp = threading.Thread(target = fft_thread, args = (F_uxp, uxp), kwargs = {'ifinverse': True})
            togrid_uyp = threading.Thread(target = fft_thread, args = (F_uyp, uyp), kwargs = {'ifinverse': True})
            togrid_uzp = threading.Thread(target = fft_thread, args = (F_uzp, uzp), kwargs = {'ifinverse': True})
            togrid_np = threading.Thread(target = fft_thread, args = (F_np, np), kwargs = {'ifinverse': True})
            togrid_uxe = threading.Thread(target = fft_thread, args = (F_uxe, uxe), kwargs = {'ifinverse': True})
            togrid_uye = threading.Thread(target = fft_thread, args = (F_uye, uye), kwargs = {'ifinverse': True})
            togrid_uze = threading.Thread(target = fft_thread, args = (F_uze, uze), kwargs = {'ifinverse': True})
            togrid_ne = threading.Thread(target = fft_thread, args = (F_ne, ne), kwargs = {'ifinverse': True})

            togrid_ax.start() ;    togrid_ay.start()  ;    togrid_az.start() 
            togrid_bx.start() ;    togrid_by.start()  # ;    togrid_bz.start() 
            togrid_uxp.start() ;    togrid_uyp.start()  ;    togrid_uzp.start() ; togrid_np.start()
            togrid_uxe.start() ;    togrid_uye.start()  ;    togrid_uze.start() ; togrid_ne.start()
            
            togrid_ax.join() ;    togrid_ay.join()  ;    togrid_az.join() 
            togrid_bx.join() ;    togrid_by.join()  # ;    togrid_bz.join() 
            togrid_uxp.join() ;    togrid_uyp.join()  ;    togrid_uzp.join() ; togrid_np.join()
            togrid_uxe.join() ;    togrid_uye.join()  ;    togrid_uze.join() ; togrid_ne.join()

            # print(abs(F_ay).max())
            # ii = input('ay')
            '''
            print("dF_ay = ", abs(dF_ay1).max(), abs(dF_ay2).max())
            print(abs(ifft(F_ay)-ay0).max())
            print(abs(ay-ay_prev).max())
            print(abs(ay).max())
            print(abs(ay_prev).max())
            # print(ifneutral, not(ifneutral))
            ii = input('a')
            '''
            
            gammap = sqrt(1.+uxp**2+uyp**2+uzp**2)
            gammae = sqrt(1.+uxe**2+uye**2+uze**2)
            
            time_togrid_end = time.time()
            togrid_time += time_togrid_end - time_togrid_start

            # ASCII output
            for k in arange(size(bx)):
                fout.write(str(t) + ' ' + str(z[k]) + ' ' + str(bx[k])+'\n')
            fout.flush()

            # check if bx has changed:
            # print("dBx = ", abs(bx-bx_prev).max())
            # ii = input("B")
            
            if ctr%picture_alias==0:
                plotting_time_start = time.time()
                plots.onthefly(z[::plotalias], (z[::plotalias]+zlen/2.+t)%zlen-zlen/2., ax0[::plotalias], ay0[::plotalias], az0[::plotalias],
                               bx0[::plotalias], by0[::plotalias], ax[::plotalias], ay[::plotalias], az[::plotalias], bx[::plotalias], by[::plotalias],
                               uxp[::plotalias], uyp[::plotalias], uzp[::plotalias], (np[::plotalias]/gammap[::plotalias],ne[::plotalias]/gammae[::plotalias]), ctr, t, omega = omega0)
                plotting_time_end = time.time()
                dplotting_time = plotting_time_end - plotting_time_start
                print("total EM energy = ", energyEM(ax, ay, az, bx, by))
                print("total particle energy = ", energyPA(np, uxp, uyp, uzp) + energyPA(ne, uxe, uye, uze) )

            elist.append(energyEM(ax, ay, az, bx, by))    ;     eplist.append(energyPA(np, uxp, uyp, uzp) + energyPA(ne, uxe, uye, uze))
            if energyPA(np, uxp, uyp, uzp) < 0.:
                print(np)
                ii = input('n')
            tlist.append(t)
            bxlist.append(copy(bx.real))
            Fbxlist.append(copy(F_bx))
            uzplist.append(copy(uzp).real) ; uzelist.append(copy(uze).real)
            nplist.append((np/gammap).real) ; nelist.append((ne/gammae).real)

            ampmax = maximum(abs(ax), abs(ay)).max()
            gammamax = maximum(abs(uyp), abs(uxp)).max()
            # tlist.append(tstore)
            # bxlist.append(bx_prev.real + ((tstore-(t-dt*0.))/dt) * (bx-bx_prev).real)
            # Fbxlist.append(F_bx_prev + ((tstore-(t-dt*0.))/dt) * (F_bx-F_bx_prev))
            # print(len(Fbxlist))
            # uzplist.append(uzp_prev.real +  ((tstore-(t-dt*0.))/dt) * (uzp-uzp_prev).real)
            # uzelist.append(uze_prev.real +  ((tstore-(t-dt*0.))/dt) * (uze-uze_prev).real)
            # nplist.append((np_prev/gammap_prev).real +  ((tstore-(t-dt))/dt) * (np/gammap - np_prev/gammap_prev).real)
            # nelist.append((ne_prev/gammae_prev).real +  ((tstore-(t-dt))/dt) * (ne/gammae - ne_prev/gammae_prev).real)
            # tstore += dtout
            ctr += 1

        # performance control:
        togrid_time += dtos1[0] + dtos2[0] + dtos3[0]
        NL_time += dtos1[1] + dtos2[1] + dtos3[1]
        diff_time += ddiff_time_end - ddiff_time_start
        plotting_time += dplotting_time
        timestep_time += timestep_end - timestep_start
        total_time = time.time() - total_time_start
        
        if toutflag and (ctr%picture_alias==0):
            print("total time = ", total_time, "s")
            print("plotting time = ", plotting_time, "s")
            print("togrid time = ", togrid_time, "s")
            print("NL time = ", NL_time, "s")
            print("timestep time = ", timestep_time, "s")
            print("hyperdiffusion time = ", diff_time, "s")            
       
    fout.close()
            
    tlist = asarray(tlist)  ;  elist = asarray(elist) ;  eplist = asarray(eplist)
    bxlist = asarray(bxlist)
    Fbxlist = asarray(Fbxlist, dtype = complex)
    uzplist = asarray(uzplist).real
    uzelist = asarray(uzelist).real
    nplist = asarray(nplist).real
    nelist = asarray(nelist).real

    nt = size(tlist)

    #    print(tlist[1]-tlist[0], dtout)
    print("dtout = ", (tlist[1:]-tlist[:-1]).min(), (tlist[1:]-tlist[:-1]).max())
    # dtout = median(tlist[1:]-tlist[:-1])
    # ii = input('T')
    # print(Fbxlist.real.max(), Fbxlist.real.min())
    #
    # bxlist_FF = fft2(bxlist)

    # should we clean the time-averaged value
    #Fbxlist_mean = Fbxlist.mean(axis = 0)
    #for k in arange(nt):
    #    Fbxlist[k, :] -= Fbxlist_mean[k]
    
    bxlist_FF = fft(Fbxlist, axis = 0) 
    ofreq = fftfreq(size(tlist), median(tlist[1:]-tlist[:-1]) / (2.*pi)) 

    # print("omega = ", ofreq)
    # print("k = ", f)
    
    bxlist_FF = fftshift(bxlist_FF)
    ofreq = fftshift(ofreq)
    f = fftshift(f)
    
    # print("wavenumber shape = ", shape(f))
    # print("frequency shape = ", shape(ofreq))
    # print(shape(bxlist_FF))
    # ii = input('T')

    #    nthalf = nt//2
    #    nzhalf = nz//2
    
    # print("omega = ", ofreq)
    # print("k = ", f / (2.*pi))
    
    # ii = input('T')
    
    # saving the data
    hio.okplane_hout(ofreq, f, bxlist_FF, hname = 'okplane_Bx.hdf', dataname = 'Bx')

    # TODO make aliases to reduce memory consumption:
    plots.show_nukeplane(omega0 = omega0, bgdfield = bbgdz)
    plots.maps(z, tlist, bxlist, (uzplist, uzelist), (nplist, nelist), ctr, zalias = 2, talias = 2)
    plots.maps_dat(zalias = 2, talias = 2)
    plots.energyplot(tlist, elist, prefix = 'EM')
    plots.energyplot(tlist, eplist, prefix = 'PA')
