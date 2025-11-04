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
from scipy.integrate import simpson

# HDF5 io:
import hio

# plotting
import plots

# auxiliary functions
import utile

# simulating a wave moving to the right along z in pair relativistic plasma
# E, B, and v are allowed to have all the three components

# physical switches:
ifmatter = False
ifonedirection = False
ifnclean = True
iflnn = False
ifgridn = False

# mesh:
nz = 1024
zlen = 50.
z = (arange(nz) / double(nz) - 0.5) * zlen
dz = z[1] - z[0]
print("dz = ", dz)

# time
dtCFL = dz * 0.1 # CFL in 1D should be not very small
dtfac = 0.1
# dtout = 0.01
ifplot = True
hdf_alias = 1000
picture_alias = 1
dtout = 0.1

# density floor
nlim = 1e-3

# initial conditions (circularly polirized wave)
ExA = 0.0
EyA = 100.0
omega0 = 10.0
f0 = omega0 / 2./ pi
tpack = sqrt(6.)
tmid = tpack * 10. # the passage of the wave through z=0
tmax = zlen 
zcenter = tpack * 4.-zlen/2.

# background magnetic field
Bxbgd = 0.0
Bybgd = 0.0
Bzbgd = 0.0

def Avec(t):
    return -sin(omega0 * t) * exp(-(t/tpack)**2/2.) / omega0

def Afield(t):
    return (cos(omega0 * t) - t/(omega0*tpack**2) * sin(omega0 * t)) * maximum(exp(-(t/tpack)**2/2.)-1e-5, 0.)

def onestep(f, F_ax, F_ay, F_az, F_bx, F_by, F_ux, F_uy, F_uz, F_n, ifmatter):
    # one RK4 step
    # avoid interference with the globals!
    
    # essential grid quantities:
    ax = ifft(F_ax) ;    ay = ifft(F_ay) #  ;    az = ifft(F_az)
    bx = ifft(F_bx) + Bxbgd ;    by = ifft(F_by) + Bybgd  # ;    bz = ifft(F_bz)
    bz = 0.
    ux = ifft(F_ux) ;    uy = ifft(F_uy) ;    uz = ifft(F_uz)
    n = ifft(F_n) # n is n gamma
    if iflnn:
        n = exp(n.real)
    # n = maximum(n, 0.)
    gamma = sqrt(1.+ux**2+uy**2+uz**2)
    vx = ux/gamma ; vy = uy/gamma ; vz = uz/gamma
    
    # Maxwell equations:
    dF_bx = 1.j * f * F_ay # one Maxwell
    dF_by = -1.j * f * F_ax # one Maxwell
    dF_ax = -1.j * f * F_by  - fft(n * vx) * complex(ifmatter) # other Maxwell
    dF_ay = 1.j * f * F_bx - fft(n * vy) * complex(ifmatter) # other Maxwell
    dF_az = 0. * n # - fft(n * uz) * complex(ifmatter) # other Maxwell

    # hydrodynamics
    dF_ux = F_ax + fft( -vz * ifft(1.j*f*F_ux) + (vy * bz - vz * by))
    dF_uy = F_ay + fft( -vz * ifft(1.j*f*F_uy) + (vz * bx - vx * bz))
    dF_uz = F_az + fft( -vz * ifft(1.j*f*F_uz) + (vx * by - vy * bx))

    if ifgridn:
        v_ext = concatenate([[vz[-1]]] +  [vz] +  [[vz[0]]])
        n_ext = concatenate([[n[-1]]] +  [n] +  [[n[0]]])
        vz_acoeff, vz_bcoeff, vz_ccoeff = utile.parcoeff(v_ext)
        n_acoeff, n_bcoeff, n_ccoeff = utile.parcoeff(n_ext) # maximum(n_ext, 0.))
        dn = (n_bcoeff * vz_acoeff + 2. * n_acoeff * vz_bcoeff) / 12. + n_bcoeff * vz_ccoeff + (vz_bcoeff * n_acoeff + 2. * vz_acoeff * n_bcoeff) / 12. + vz_bcoeff * n_ccoeff
        dn = -dn/dz
        dF_n = fft(dn)
        #print("hspaes = ", shape(n), shape(n_ext), shape(dn))
        #ii = input("h")
    else:
        if iflnn:
            dF_n = fft(ifft(-1.j * f * fft(n * vz)) / n)
        else:
            dF_n = -1.j * f * fft(vz * n) #
        dn = ifft(dF_n)
    
    dt_n = 0.5/maximum(abs(dn).max(), 1./tmax)    
    
    return dF_ax, dF_ay, dF_az, dF_bx, dF_by, dF_ux, dF_uy, dF_uz, dF_n, dt_n

def sewerrun():
    
    f = fftfreq(nz, d = dz / (2. * pi)) # Fourier mesh

    # initial conditions:
    ax0 = ExA * Afield(z-zcenter)
    ay0 = EyA * Afield(z-zcenter)
    bx0 = -EyA * Afield(z-zcenter)
    by0 = ExA * Afield(z-zcenter)
    az0 = z * 0. ;   bz0 = z * 0.
    az = az0
    # 4-velocity
    ux0 = 0.*z  ;    uy0 = EyA * Avec(z-zcenter) 
    uz0 = uy0**2/2.
    n0 = ones(nz) * 1.0 # density ; let us keep it unity, meaning time is in omega_p units. Lengths are internally in c/f = 2pi c / omega units, that allows a simpler expression for d/dz 
    if iflnn:
        n0 = log(n0)
    
    # Fourier images
    F_bx = fft(bx0)
    F_ax = fft(ax0)
    F_ux = fft(ux0)
    F_by = fft(by0)
    F_ay = fft(ay0)
    F_uy = fft(uy0)
    # F_bz = fft(bz0)
    F_az = fft(az0)
    F_uz = fft(uz0)
    F_n  = fft(n0)

    if ifonedirection:
        # the square root is the correction for plasma dispersion
        F_ay[abs(f)>0.] = 1. * copy(F_bx)[abs(f)>0.] * sqrt(1.+(2.*pi*f)**(-2))[abs(f)>0.]
        F_ax[abs(f)>0.] = -1. * copy(F_by)[abs(f)>0.]  * sqrt(1.+(2.*pi*f)**(-2))[abs(f)>0.]

    dt = dtCFL 
    t = 0. 
    ctr = 0
    tstore = 0.
    tlist = []
    bxlist = []
    Fbxlist = []
    uzlist = []
    uylist = []
    nlist = []
    mlist = [] # particle mass
    paelist = [] # particle energy
    emelist = [] # EM fields energy    
    
    # hyperdiffusion core
    fsq = (f * conj(f)).real
    hyperpower = 2.0
    # hypercore = exp(-(fsq / (fsq.real).max())**hyperpower * 0.5) + 0.j #  * (2.*pi)**2
    hyperpower = 2.0
    cutofffactor = 2.0
    hypershift = 0.0
    hypercore = minimum(exp(hypershift-(fsq / (fsq.max() * cutofffactor))**hyperpower * dt), 1.0) + 0.j
    hypercore_n = minimum(exp(hypershift-(fsq / (fsq.max() * cutofffactor))**hyperpower * dtCFL), 1.0) + 0.j
    # print(f)

    fout_energy = open('sewer_energy.dat', 'w+')
    fout = open('sewerout.dat', 'w+')
    fout.write('# t -- z -- Bx \n')
    fout_energy.write('# t -- mass -- EM energy -- PA energy\n')
    
    while t < tmax:        
        if t > (tstore + dtout - dt):
            # save previous values
            F_bx_prev = F_bx
            ax_prev = ifft(F_ax) ;    ay_prev = ifft(F_ay)  # ;    az_prev = ifft(F_az)
            bx_prev = ifft(F_bx) ;    by_prev = ifft(F_by)  # ;    bz = ifft(F_bz)
            ux_prev = ifft(F_ux) ;    uy_prev = ifft(F_uy)  ;    uz_prev = ifft(F_uz)
            n_prev = ifft(F_n)
            gamma_prev = sqrt(1.+ux_prev**2+uy_prev**2+uz_prev**2)
            
        # TODO: make it dictionaries or structures    
        dF_ax1, dF_ay1, dF_az1, dF_bx1, dF_by1, dF_ux1, dF_uy1, dF_uz1, dF_n1, dt_n = onestep(f, F_ax, F_ay, F_az, F_bx, F_by, F_ux, F_uy, F_uz, F_n, ifmatter)
        # adaptive time step:
        duz = ifft(dF_uz1) ; duy = ifft(dF_uy1) ; dn = ifft(dF_n1)
        dt_uy = 1./abs(duy).max()
        dt_uz = 1./abs(duz).max()
        #        dt_n = tmax # !!!temporary 1./abs(dn).max()
        dt = minimum(dtCFL, minimum(minimum(dt_uz, dt_uy), dt_n) * dtfac)
        # print("dt = ", dtCFL, ", ", dt_uz*0.01, ", ", dt_uy * 0.01, ", ", dt_n * 0.01)
        
        dF_ax2, dF_ay2, dF_az2, dF_bx2, dF_by2, dF_ux2, dF_uy2, dF_uz2, dF_n2, dt_n = onestep(f, F_ax + dF_ax1/2. * dt, F_ay  + dF_ay1/2. * dt, dF_az1/2. * dt, F_bx + dF_bx1/2. * dt, F_by + dF_by1/2. * dt, F_ux + dF_ux1/2. * dt, F_uy + dF_uy1/2. * dt, F_uz  + dF_uz1/2. * dt, F_n  + dF_n1/2. * dt, ifmatter)
        dF_ax3, dF_ay3, dF_az3, dF_bx3, dF_by3, dF_ux3, dF_uy3, dF_uz3, dF_n3, dt_n = onestep(f, F_ax + dF_ax2/2. * dt, F_ay  + dF_ay2 /2. * dt, dF_az2 / 2. * dt, F_bx + dF_bx2 / 2. * dt, F_by + dF_by2 / 2. * dt, F_ux + dF_ux2 / 2. * dt, F_uy + dF_uy2 / 2. * dt, F_uz  + dF_uz2 / 2. * dt, F_n  + dF_n2 / 2. * dt, ifmatter)    
        dF_ax4, dF_ay4, dF_az4, dF_bx4, dF_by4, dF_ux4, dF_uy4, dF_uz4, dF_n4, dt_n = onestep(f, F_ax + dF_ax3 * dt, F_ay  + dF_ay3 * dt, dF_az3 * dt, F_bx + dF_bx3 * dt, F_by + dF_by3 * dt, F_ux + dF_ux3 * dt, F_uy + dF_uy3 * dt, F_uz  + dF_uz3 * dt, F_n  + dF_n3 * dt, ifmatter)    
        # time step:
        F_bx += (dF_bx1 + dF_bx2 * 2. + dF_bx3 * 2. + dF_bx4) / 6. * dt
        F_by += (dF_by1 + dF_by2 * 2. + dF_by3 * 2. + dF_by4) / 6. * dt
        F_ax += (dF_ax1 + dF_ax2 * 2. + dF_ax3 * 2. + dF_ax4) / 6. * dt
        F_ay += (dF_ay1 + dF_ay2 * 2. + dF_ay3 * 2. + dF_ay4) / 6. * dt 
        # F_az += (dF_az1 + dF_az2 * 2. + dF_az3 * 2. + dF_az4) / 6. * dt
        F_ux += (dF_ux1 + dF_ux2 * 2. + dF_ux3 * 2. + dF_ux4) / 6. * dt
        F_uy += (dF_uy1 + dF_uy2 * 2. + dF_uy3 * 2. + dF_uy4) / 6. * dt
        F_uz += (dF_uz1 + dF_uz2 * 2. + dF_uz3 * 2. + dF_uz4) / 6. * dt
        F_n += (dF_n1 + dF_n2 * 2. + dF_n3 * 2. + dF_n4) / 6. * dt
        t += dt

        # dyperdiffusion:
        F_bx *= hypercore ;   F_by *= hypercore
        F_ax *= hypercore ;   F_ay *= hypercore  # ;   F_az *= hypercore 
        F_ux *= hypercore ;   F_uy *= hypercore  ;   F_uz *= hypercore   ;    F_n *= hypercore

        if ifnclean and not(iflnn):
            n = maximum(ifft(F_n), nlim)
            F_n = fft(n) * hypercore_n
        
        if t > (tstore + dtout):
            print("t = ", t)

            if ctr%picture_alias==0:
                # Fourier spectrum:
                plots.fourier(f/2./pi, F_bx, f0, ctr)

            # print(F_bx.real.max(), F_bx.real.min())
            ax = ifft(F_ax) ;    ay = ifft(F_ay)  # ;    az = ifft(F_az)
            bx = ifft(F_bx) ;    by = ifft(F_by)  # ;    bz = ifft(F_bz)
            ux = ifft(F_ux) ;    uy = ifft(F_uy)  ;    uz = ifft(F_uz)
            n = ifft(F_n)
            if iflnn:
                n = exp(n)
            gamma = sqrt(1.+ux**2+uy**2+uz**2)
            # vx = ux/gamma ; vy = uy/gamma ; vz = uz/gamma
            
            # print("Bx = ", bx.min(), '..', bx.max())
            # print("Ey = ", ay.min(), '..', ay.max())
            print("dt = ", dt)

            # HDF5 output:
            if ctr == 0:
                hout = hio.fewout_init('sout.hdf5',
                                       {"ifmatter": ifmatter, "ExA": ExA, "EyA": EyA,
                                        "omega0": omega0, "tpack": tpack, "tmid": tmid, "Bz": Bzbgd, "Bx": Bxbgd},
                                       z, zhalf = z)

            hio.fewout_dump(hout, ctr, t, (ax, ay), (bx+Bxbgd, by+Bybgd), (ux, uy, uz), n)
            
            # ASCII output
            for k in arange(size(bx)):
                fout.write(str(t) + ' ' + str(z[k]) + ' ' + str(bx[k])+'\n')
            fout.flush()

            mtot = simpson(n.real, x = z)
            epatot = simpson((n*(gamma-1.)).real, x = z)
            emetot = (simpson(bx.real**2+by.real**2, x = z) + simpson(ax.real**2+ay.real**2, x = z))/2.

            fout_energy.write(str(t) + ' ' + str(mtot) + ' ' + str(emetot) + ' ' + str(epatot) + '\n')
            fout_energy.flush()
            
            # tlist.append(t)
            mlist.append(mtot)
            paelist.append(epatot)
            emelist.append(emetot)            
            
            if ctr%picture_alias==0:
                plots.onthefly(z, (z+zlen/2.+t)%zlen-zlen/2., ax0, ay0, az0, bx0 + Bxbgd, by0 + Bybgd, ax, ay, az, bx + Bxbgd, by + Bybgd, ux, uy, uz, n/gamma, ctr, t)
                
            tlist.append(tstore)
            bxlist.append(bx_prev.real + ((tstore-(t-dt))/dt) * (bx-bx_prev).real + Bxbgd)
            Fbxlist.append(F_bx_prev + ((tstore-(t-dt))/dt) * (F_bx-F_bx_prev))
            # print(len(Fbxlist))
            uylist.append(uy_prev.real +  ((tstore-(t-dt))/dt) * (uy-uy_prev).real)
            uzlist.append(uz_prev.real +  ((tstore-(t-dt))/dt) * (uz-uz_prev).real)
            nlist.append((n_prev/gamma_prev).real +  ((tstore-(t-dt))/dt) * (n/gamma - n_prev/gamma_prev).real)
            tstore += dtout
            ctr += 1

    fout.close()   ; fout_energy.close()
    hout.close()
    
    tlist = asarray(tlist)
    bxlist = asarray(bxlist)
    Fbxlist = asarray(Fbxlist, dtype = complex)
    uylist = asarray(uylist).real
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

    plots.show_nukeplane(omega0 = omega0)
    
    plots.maps(z, tlist, bxlist, uylist, uzlist, nlist, ctr, zalias = 2, talias = 1)

    # final mass and energy plots
    if ifplot:
        clf()
        plot(tlist, mlist, 'k.')
        xlabel(r'$t$')  ;  ylabel(r'$M_{\rm tot}$')
        savefig('m.png')
        clf()
        plot(tlist, emelist, 'k.', label = 'EM')
        plot(tlist, paelist, 'rx', label = 'particles')
        plot(tlist, paelist+emelist, 'g--', label = 'total')
        ylim((paelist+emelist).max()*1e-5, (paelist+emelist).max()*2.)
        yscale('log')
        legend()
        xlabel(r'$t$')  ;  ylabel(r'$E$')
        savefig('e.png')
    
    
# ffmpeg -f image2 -r 20 -pattern_type glob -i 'EB*.png' -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  -pix_fmt yuv420p -b 8192k EB.mp4
# uGOcompare('sout_A2_nofeed.hdf5', arange(1000))
