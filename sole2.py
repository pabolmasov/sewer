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
nvars = 14 # three magnetic fields, three electric fields, three velocities + density for two particle species.

# mesh:
nz = 4096
zlen = 100.
z = (arange(nz) / double(nz) - 0.5) * zlen
dz = z[1] - z[0]
print("dz = ", dz)

# time
# t = 0.
dt = dz * 0.5 # CFL in 1D should be not very small
tmax = 200.
dtout = 0.01
picture_alias = 10


# initial conditions (circularly polirized wave)
z0 = 5.0
f0 = 0.5
amp0 = 0.01
bbgdx = 0.0  ; bbgdy = 0.0 ; bbgdz = 0.0
bx0 = sin(2. * pi * f0 * z) * exp(-(z/z0)**6/2.) * amp0 + bbgdx
by0 =  - cos(2. * pi * f0 * z) * exp(-(z/z0)**6/2.) * amp0 * 0. + bbgdy
bz0 = z *  0. + bbgdz
bz = bz0

ax0 = cos(2. * pi * f0 * z) * exp(-(z/z0)**6/2.) * amp0 * 0.
ay0 =  sin(2. * pi * f0 * z) * exp(-(z/z0)**6/2.) * amp0
az0 = z * 0.
# 4-velocity
ux0p = 0. * z
uy0p = 0. * z
uz0p = 0. * z
n0p = ones(nz) * 1.0
ux0e = 0. * z
uy0e = 0. * z
uz0e = 0. * z
n0e = ones(nz) * 1.0
# density ; let us keep it unity, meaning time is in omega_p units. Lengths are internally in c/f = 2pi c / omega units, that allows a simpler expression for d/dz 



def fft_thread(inar, outar, ifinverse = False):
    if ifinverse:
        outar = ifft(inar)
    else:
        outar = fft(inar)

def NL_thread(f, vz, Fui, addgrid, outar):
    outar += fft( vz * ifft(1.j*f*Fui)  + addgrid)
        
def onestep(f, F_ax, F_ay, F_az, F_bx, F_by, F_uxp, F_uyp, F_uzp, F_np, F_uxe, F_uye, F_uze, F_ne, ifmatter):
    # one RK4 step
    # avoid interference with the globals!
    
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
    gammae = sqrt(1.+uxe**2+uye**2+uze**2)
    vxe = uxe/gammae ; vye = uye/gammae ; vze = uze/gammae

    # currents:
    jx = np * uxp - ne * uxe
    jy = np * uyp - ne * uye
    jz = np * uzp - ne * uze
    
    # Maxwell equations:
    dF_bx = -1.j * f * F_ay # one Maxwell
    dF_by = 1.j * f * F_ax # one Maxwell
    dF_ax = 1.j * f * F_by  - fft(jx) * complex(ifmatter) # other Maxwell
    dF_ay = -1.j * f * F_bx - fft(jy) * complex(ifmatter) # other Maxwell
    dF_az =   - fft(jz) * complex(ifmatter) # other Maxwell

    #
    # NL_thread(f, vz, Fui, addgrid, outar)

    # hydrodynamics
    dF_uxp = F_ax #+ fft( -vzp * ifft(-1.j*f*F_uxp) + vyp * bz - vzp * by)
    dF_uyp = F_ay #+ fft( -vzp * ifft(-1.j*f*F_uyp) + vzp * bx - vxp * bz)
    dF_uzp = F_az #+ fft( -vzp * ifft(-1.j*f*F_uzp) + vxp * by - vyp * bx)
    dF_uxe = -F_ax #+ fft( -vzp * ifft(-1.j*f*F_uxp) + vyp * bz - vzp * by)
    dF_uye = -F_ay #+ fft( -vzp * ifft(-1.j*f*F_uyp) + vzp * bx - vxp * bz)
    dF_uze = -F_az #+ fft( -vzp * ifft(-1.j*f*F_uzp) + vxp * by - vyp * bx)

    time_NL_start = time.time()
    # adding non-linear terms (parallelized)
    NLxp_thread = threading.Thread(target = NL_thread, args = (f, vzp, F_uxp, vyp * bz - vzp * by, dF_uxp))
    NLyp_thread = threading.Thread(target = NL_thread, args = (f, vzp, F_uyp, vzp * bx - vxp * bz, dF_uyp))
    NLzp_thread = threading.Thread(target = NL_thread, args = (f, vzp, F_uzp, vxp * by - vyp * bx, dF_uzp))
    NLxe_thread = threading.Thread(target = NL_thread, args = (f, vze, F_uxe, vye * bz - vze * by, dF_uxe))
    NLye_thread = threading.Thread(target = NL_thread, args = (f, vze, F_uye, vze * bx - vxe * bz, dF_uye))
    NLze_thread = threading.Thread(target = NL_thread, args = (f, vze, F_uze, vxe * by - vye * bx, dF_uze))

    NLxp_thread.start() ; NLyp_thread.start() ; NLzp_thread.start() ;     NLxe_thread.start() ; NLye_thread.start() ; NLze_thread.start()

    NLxp_thread.join() ; NLyp_thread.join() ; NLzp_thread.join() ;     NLxe_thread.join() ; NLye_thread.join() ; NLze_thread.join()
    
    time_NL_end = time.time()
    NL_time_diff = time_NL_end - time_NL_start
    
    dF_np = 1.j * f * fft(vzp*np) 
    #    dF_uxe = -F_ax + fft( -vze * ifft(-1.j*f*F_uxe) + vye * bz - vze * by)
    #    dF_uye = -F_ay + fft( -vze * ifft(-1.j*f*F_uye) + vze * bx - vxe * bz)
    #    dF_uze = -F_az + fft( -vze * ifft(-1.j*f*F_uze) + vxe * by - vye * bx)
    dF_ne = 1.j * f * fft(vze * ne) 

    return dF_ax, dF_ay, dF_az, dF_bx, dF_by, dF_uxp, dF_uyp, dF_uzp, dF_np, dF_uxe, dF_uye, dF_uze, dF_ne, (togrid_time_diff, NL_time_diff)

def sewerrun2():

    # performance control:
    togrid_time = 0.
    NL_time = 0.0
    
    f = fftfreq(nz, d = dz / (2. * pi)) # Fourier mesh

    # Fourier images
    F_bx = fft(bx0)
    F_ax = fft(ax0)
    F_uxp = fft(ux0p)
    F_uxe = fft(ux0e)
    F_by = fft(by0)
    F_ay = fft(ay0)
    F_uyp = fft(uy0p)
    F_uye = fft(uy0e)
    F_bz = fft(bz0)
    F_az = fft(az0)
    F_uzp = fft(uz0p)
    F_uze = fft(uz0e)
    F_np  = fft(n0p)
    F_ne  = fft(n0e)

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
    uzplist = [] ; uzelist = []
    nplist = [] ;   nelist = []
    
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
            uxp_prev = ifft(F_uxp) ;    uyp_prev = ifft(F_uyp)  ;    uzp_prev = ifft(F_uzp)
            np_prev = ifft(F_np) ; gammap_prev = sqrt(1.+uxp_prev**2+uyp_prev**2+uzp_prev**2)
            uxe_prev = ifft(F_uxe) ;    uye_prev = ifft(F_uye)  ;    uze_prev = ifft(F_uze)
            ne_prev = ifft(F_ne) ; gammae_prev = sqrt(1.+uxe_prev**2+uye_prev**2+uze_prev**2)
           
        # TODO: make it dictionaries or structures
    
        dF_ax1, dF_ay1, dF_az1, dF_bx1, dF_by1, dF_uxp1, dF_uyp1, dF_uzp1, dF_np1, dF_uxe1, dF_uye1, dF_uze1, dF_ne1, dtos1 = onestep(f, F_ax, F_ay, F_az, F_bx, F_by, F_uxp, F_uyp, F_uzp, F_np, F_uxe, F_uye, F_uze, F_ne, ifmatter)
        dF_ax2, dF_ay2, dF_az2, dF_bx2, dF_by2, dF_uxp2, dF_uyp2, dF_uzp2, dF_np2, dF_uxe2, dF_uye2, dF_uze2, dF_ne2, dtos2 = onestep(f, F_ax + dF_ax1/3. * dt, F_ay  + dF_ay1/3. * dt, dF_az1/3. * dt, F_bx + dF_bx1/3. * dt, F_by + dF_by1/3. * dt, F_uxp + dF_uxp1/3. * dt, F_uyp + dF_uyp1/3. * dt, F_uzp  + dF_uzp1/3. * dt, F_np  + dF_np1/3. * dt, F_uxe + dF_uxe1/3. * dt, F_uye + dF_uye1/3. * dt, F_uze  + dF_uze1/3. * dt, F_ne  + dF_ne1/3. * dt, ifmatter)
        dF_ax2, dF_ay2, dF_az2, dF_bx2, dF_by2, dF_uxp2, dF_uyp2, dF_uzp2, dF_np2, dF_uxe2, dF_uye2, dF_uze2, dF_ne2, dtos3 = onestep(f, F_ax + dF_ax2 * 2./3. * dt, F_ay  + dF_ay2 * 2./3. * dt, dF_az2 * 2./3. * dt, F_bx + dF_bx2 * 2./3. * dt, F_by + dF_by2 * 2./3. * dt, F_uxp + dF_uxp2 * 2./3. * dt, F_uyp + dF_uyp2 * 2./3. * dt, F_uzp  + dF_uzp2 * 2./3. * dt, F_np  + dF_np2 * 2./3. * dt, F_uxe + dF_uxe2 * 2./3. * dt, F_uye + dF_uye2 * 2./3. * dt, F_uze  + dF_uze2 * 2./3. * dt, F_ne  + dF_ne2 * 2./3. * dt, ifmatter)    
   
        # time step:
        F_bx += (dF_bx1 * 0.25 + dF_bx2 * 0.75) * dt ;    F_by += (dF_by1 * 0.25 + dF_by2 * 0.75) * dt
        F_ax += (dF_ax1 * 0.25 + dF_ax2 * 0.75) * dt ;    F_ay += (dF_ay1 * 0.25 + dF_ay2 * 0.75) * dt ;    F_az += (dF_az1 * 0.25 + dF_az2 * 0.75) * dt
        F_uxp += (dF_uxp1 * 0.25 + dF_uxp2 * 0.75) * dt ;    F_uyp += (dF_uyp1 * 0.25 + dF_uyp2 * 0.75) * dt ;    F_uzp += (dF_uzp1 * 0.25 + dF_uzp2 * 0.75) * dt
        F_np += (dF_np1 * 0.25 + dF_np2 * 0.75) * dt
        F_uxe += (dF_uxe1 * 0.25 + dF_uxe2 * 0.75) * dt ;    F_uye += (dF_uye1 * 0.25 + dF_uye2 * 0.75) * dt ;    F_uze += (dF_uze1 * 0.25 + dF_uze2 * 0.75) * dt
        F_ne += (dF_ne1 * 0.25 + dF_ne2 * 0.75) * dt
        t += dt

        # dyperdiffusion:
        F_bx *= hypercore ;   F_by *= hypercore
        F_ax *= hypercore ;   F_ay *= hypercore  ;   F_az *= hypercore 
        F_uxp *= hypercore ;   F_uyp *= hypercore  ;   F_uzp *= hypercore   ;    F_np *= hypercore
        F_uxe *= hypercore ;   F_uye *= hypercore  ;   F_uze *= hypercore   ;    F_ne *= hypercore

        # performance control:
        togrid_time += dtos1[0] + dtos2[0] + dtos3[0]
        NL_time += dtos1[1] + dtos2[1] + dtos3[1]
        
        if t > (tstore + dtout):
            print("t = ", t)
            print("togrid_time = ", togrid_time)
            print("NL_time = ", NL_time)
            togrid_time = 0.  ;  NL_time = 0.
            
            if ctr%picture_alias==0:
                # Fourier spectrum:
                clf()
                plot(f/2./pi, F_bx.real, 'k.')
                plot(f/2./pi, F_bx.imag, 'gx')
                plot([f0, f0], [0.,sqrt(F_bx.real**2 + F_bx.imag**2).max()], 'r-')
                plot([-f0, -f0], [0.,sqrt(F_bx.real**2 + F_bx.imag**2).max()], 'r-')
                xlim(-2.*f0, 2.* f0)
                xlabel(r'$f$')  ;   ylabel(r'$\tilde b_x$')
                savefig('f{:05d}.png'.format(ctr))

            # print(F_bx.real.max(), F_bx.real.min())
            ax = ifft(F_ax) ;    ay = ifft(F_ay)  ;    az = ifft(F_az)
            bx = ifft(F_bx) ;    by = ifft(F_by)  # ;    bz = ifft(F_bz)
            uxp = ifft(F_uxp) ;    uyp = ifft(F_uyp)  ;    uzp = ifft(F_uzp)
            np = ifft(F_np) ;       gammap = sqrt(1.+uxp**2+uyp**2+uzp**2)
            uxe = ifft(F_uxe) ;    uye = ifft(F_uye)  ;    uze = ifft(F_uze)
            ne = ifft(F_ne)  ;       gammae = sqrt(1.+uxe**2+uye**2+uze**2)
            # vx = ux/gamma ; vy = uy/gamma ; vz = uz/gamma
            
            print("Bx = ", bx.min(), '..', bx.max())
            print("Ey = ", ay.min(), '..', ay.max())

            # ASCII output
            for k in arange(size(bx)):
                fout.write(str(t) + ' ' + str(z[k]) + str(bx[k])+'\n')
            fout.flush()
            
            if ctr%picture_alias==0:
                plots.onthefly(z, (z+zlen/2.+t)%zlen-zlen/2., ax0, ay0, az0, bx0, by0, ax, ay, az, bx, by, uxp, uyp, uzp, (np/gammap,ne/gammae), ctr, t)
                
            tlist.append(tstore)
            bxlist.append(bx_prev.real + ((tstore-(t-dt))/dt) * (bx-bx_prev).real)
            Fbxlist.append(F_bx_prev + ((tstore-(t-dt))/dt) * (F_bx-F_bx_prev))
            # print(len(Fbxlist))
            uzplist.append(uzp_prev.real +  ((tstore-(t-dt))/dt) * (uzp-uzp_prev).real)
            uzelist.append(uze_prev.real +  ((tstore-(t-dt))/dt) * (uze-uze_prev).real)
            nplist.append((np_prev/gammap_prev).real +  ((tstore-(t-dt))/dt) * (np/gammap - np_prev/gammap_prev).real)
            nelist.append((ne_prev/gammae_prev).real +  ((tstore-(t-dt))/dt) * (ne/gammae - ne_prev/gammae_prev).real)
            tstore += dtout
            ctr += 1

    fout.close()
            
    tlist = asarray(tlist)
    bxlist = asarray(bxlist)
    Fbxlist = asarray(Fbxlist, dtype = complex)
    uzplist = asarray(uzplist).real
    uzelist = asarray(uzelist).real
    nplist = asarray(nplist).real
    nelist = asarray(nelist).real

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

    plots.show_nukeplane(f0 = f0, bgdfield = bbgdz)
    plots.maps(z, tlist, bxlist, (uzplist, uzelist), (nplist, nelist), ctr)
 
