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
from scipy.interpolate import interp1d

# HDF5 io:
import hio

# plotting
import plots

# auxiliary functions
import utile

# simulating a wave moving to the right along z in pair relativistic plasma
# E, B, and v are allowed to have all the three components

# physical switches:
ifmatter = True
ifonedirection = False
ifnclean = True
iflnn = False
ifgridn = False
ifsource = False
ifvdamp = False

# mesh:
nz = 4096
zlen = 100.
z0 = (arange(nz) / double(nz) - 0.5) * zlen
dz = z0[1] - z0[0]
print("dz = ", dz)

# time
dtCFL = dz * 0.1 # CFL in 1D should be not very small
dtfac = 0.1
# dtout = 0.01
ifplot = True
hdf_alias = 1000
picture_alias = 10
dtout = 0.1

# density floor
nlim = 1e-3

# decay parameters
dtcay = dtCFL * 1.0
dzcay = 5.0

# initial conditions (circularly polirized wave)
ExA = 0.0
EyA = 20.0
omega0 = 10.0
f0 = omega0 / 2./ pi
tpack = sqrt(6.)
tmid = tpack * 5. # the passage of the wave through z=0
tmax = zlen * 0.5
zcenter = tpack * 4.-zlen/2.
zbuffer = 2.0

# background magnetic field
Bxbgd = 0.0
Bybgd = 0.0
Bzbgd = 0.0

def Avec(t):
    return -sin(omega0 * t) * exp(-(t/tpack)**2/2.) / omega0

def Afield(t):
    return (cos(omega0 * t) - t/(omega0*tpack**2) * sin(omega0 * t)) * exp(-(t/tpack)**2/2.)

def jfield(t):
    return -(2. * t / tpack**2 * cos(omega0 * t) + (omega0 + 1./omega0 / tpack**2 - (t/tpack)**2/omega0) * sin(omega0 * t)) * exp(-(t/tpack)**2/2.)

def jacoden(z, n0):
    # density as a Jacobian (assuming n0 = 1.0)
    return dz * n0 / maximum(abs(roll(z, 1)- z) % (zlen+dz), abs(z -roll(z, -1)) % (zlen+dz))

def monotonic_split(x):

    n = size(x)
    
    xmon = []

    currentlist = [0]
    
    for k in arange(n-1, dtype = int):
        if (x[k] - x[k-1]) * (x[k+1]-x[k]) > 0.:
            currentlist.append(k+1)
        else:
            xmon.append(asarray(currentlist))
            currentlist = [k+1]

    xmon.append(asarray(currentlist))
            
    return xmon
    
    
def onestep(f, F_ax, F_ay, F_bx, F_by, ux, uy, uz, n0, z, t):
    # one RK4 step

    znorm = (z-z0.min()) % (z0.max() - z0.min())+z0.min()
    
    # essential grid quantities:
    ax = ifft(F_ax) ;    ay = ifft(F_ay) 
    bx = ifft(F_bx) + Bxbgd ;    by = ifft(F_by) + Bybgd  # ;    bz = ifft(F_bz)
    bz = 0.
    # ux = ifft(F_ux) ;    uy = ifft(F_uy) ;    uz = ifft(F_uz)
    n =  jacoden(z, n0) # lab-frame density n\gamma

    gamma = sqrt(1.+ux**2+uy**2+uz**2)
    vx = ux/gamma ; vy = uy/gamma ; vz = uz/gamma

    # two mappings
    if ifmatter:
        # mapping currents from Lagrangian to Eulerian
        wg = z[1:] <= z[:-1]
        if wg.sum() <= 0:
            jxfun = interp1d(znorm, -n * vx, bounds_error = False, fill_value = 0.) 
            jyfun = interp1d(znorm, -n * vy, bounds_error = False, fill_value = 0.)
            jx = jxfun(z0) ; jy = jyfun(z0)
        else:
            # there are non-monotonic regions
            wmonotonic = monotonic_split(znorm)
            nchunks = len(wmonotonic)
            jx = z * 0. ;  jy = z * 0.
            # print(nchunks, " monotonic chunks, sizes: ")
            for k in arange(nchunks):
                w = (z0 > znorm[wmonotonic[k]].min()) * (z0 < znorm[wmonotonic[k]].max())
                if w.sum() > 2:
                    jxfun = interp1d(znorm[wmonotonic[k]], -(n * vx)[wmonotonic[k]], bounds_error = False, fill_value = 0.) 
                    jyfun = interp1d(znorm[wmonotonic[k]], -(n * vy)[wmonotonic[k]], bounds_error = False, fill_value = 0.)
                    jx[w] += jxfun(z0[w]) ; jy[w] += jyfun(z0[w])
                    # print("size ", w.sum())
                    
    # mapping fields from Eulerian grid to Lagransian
    ax = ifft(F_ax).real ;   ay = ifft(F_ay).real
    bx = ifft(F_bx).real ;   by = ifft(F_by).real
    axfun = interp1d(z0, ax, bounds_error = False, fill_value = (ax[-1], ax[0]))
    ayfun = interp1d(z0, ay, bounds_error = False, fill_value = (ay[-1], ay[0]))
    bxfun = interp1d(z0, bx + Bxbgd, bounds_error = False, fill_value = (bx[-1] + Bxbgd, bx[0] + Bxbgd))
    byfun = interp1d(z0, by + Bybgd, bounds_error = False, fill_value = (by[-1] + Bybgd, by[0] + Bybgd))
    
    # Maxwell equations:
    dF_bx = 1.j * f * F_ay # one Maxwell
    dF_by = -1.j * f * F_ax 
    dF_ax = -1.j * f * F_by # other Maxwell 
    dF_ay = 1.j * f * F_bx 
    # dF_az = 0. * n # - fft(n * uz) * complex(ifmatter) # other Maxwell

    if ifmatter:
        dF_ax += fft(jx) 
        dF_ay += fft(jy) 

    if ifsource:
        sourceshape = exp(-(z-zcenter)**2/tpack / 2.)
        dF_ax += ExA * jfield(t - tmid) * sourceshape
        dF_ay += EyA * jfield(t - tmid) * sourceshape
        dF_bx += -EyA * jfield(t - tmid) * sourceshape
        dF_by += ExA * jfield(t - tmid) * sourceshape
        
        
    # hydrodynamics
    # dF_ux = F_ax + fft( -vz * ifft(1.j*f*F_ux) + (vy * bz - vz * by))
    # dF_uy = F_ay + fft( -vz * ifft(1.j*f*F_uy) + (vz * bx - vx * bz))
    # dF_uz = F_az + fft( -vz * ifft(1.j*f*F_uz) + (vx * by - vy * bx))
    dux = axfun(znorm) + vy * Bzbgd - vz * byfun(znorm)
    duy = ayfun(znorm) + vz * bxfun(znorm) - vx * Bzbgd
    duz =  vx * byfun(znorm) - vy * bxfun(znorm)    
    
    dzz = vz 
    
    # dt_n = 0.5/maximum(abs(dn).max(), 1./tmax)    
    
    return dF_ax, dF_ay, dF_bx, dF_by, dux, duy, duz, dzz # , dt_n

def sewerrun():
    
    f = fftfreq(nz, d = dz / (2. * pi)) # Fourier mesh
    z = copy(z0)

    # initial conditions:
    ax0 = ExA * Afield(z-zcenter)
    ay0 = EyA * Afield(z-zcenter)
    bx0 = -EyA * Afield(z-zcenter)
    by0 = ExA * Afield(z-zcenter)
    az0 = z * 0. # ;   bz = z * 0.
    az = az0
    # 4-velocity
    ux = 0.*z  ;    uy = -EyA * Avec(z-zcenter) 
    uz = uy**2/2.
    n0 = ones(nz) * 1.0 # * (z0 > z0.min()+zbuffer) # density ; let us keep it unity, meaning time is in omega_p units. Lengths are internally in c/f = 2pi c / omega units, that allows a simpler expression for d/dz 

    if ifsource:
        ax0 *= 0. ; ay0 *= 0.
        bx0 *= 0. ; by0 *= 0.
        uy *= 0. ;  uz *= 0.
    if Bxbgd > 0.1:
        uy *= 0. ; uz *= 0.
        
    # Fourier images
    F_bx = fft(bx0)
    F_ax = fft(ax0)
    # F_ux = fft(ux0)
    F_by = fft(by0)
    F_ay = fft(ay0)
    # F_uy = fft(uy0)
    # F_bz = fft(bz0)
    # F_az = fft(az0)
    #F_uz = fft(uz0)
    # F_n  = fft(n0)

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
    # hypercore = exp(-(fsq / (fsq.real).max())**hyperpower * 0.5) + 0.j #  * (2.*pi)**2
    hyperpower = 2.0
    cutofffactor = 2.0
    hypershift = 0.0
    hypercore = minimum(exp(hypershift-(fsq / (fsq.max() * cutofffactor))**hyperpower * dt), 1.0) + 0.j
    # hypercore_n = minimum(exp(hypershift-(fsq / (fsq.max() * cutofffactor))**hyperpower * dtCFL), 1.0) + 0.j
    # print(f)

    fout_energy = open('slew_energy.dat', 'w+')
    fout = open('slewout.dat', 'w+')
    fout.write('# t -- z -- Bx \n')
    fout_energy.write('# t -- mass -- EM energy -- PA energy\n')

    plots.slew(0., z0, z, ifft(F_ay).real, ifft(F_bx).real, uy, uz, n0, -1)
    #     ii = input('p')
    
    while t < tmax:        
        if t > (tstore + dtout - dt):
            # save previous values
            F_bx_prev = F_bx
            ax_prev = ifft(F_ax) ;    ay_prev = ifft(F_ay)  # ;    az_prev = ifft(F_az)
            bx_prev = ifft(F_bx) ;    by_prev = ifft(F_by)  # ;    bz = ifft(F_bz)
            ux_prev = ux ;    uy_prev = uy  ;    uz_prev = uz
            n_prev = jacoden(z, n0)
            gamma_prev = sqrt(1.+ux_prev**2+uy_prev**2+uz_prev**2)
            
        # TODO: make it dictionaries or structures    
        dF_ax1, dF_ay1, dF_bx1, dF_by1, dux1, duy1, duz1, dzz1 = onestep(f, F_ax, F_ay, F_bx, F_by, ux, uy, uz, n0, z, t)
        # adaptive time step:
        # duz = ifft(dF_uz1) ; duy = ifft(dF_uy1) ; dn = ifft(dF_n1)
        dt_uy = 1./abs(duy1).max()
        dt_uz = 1./abs(duz1).max()
        #        dt_n = tmax # !!!temporary 1./abs(dn).max()
        dt = minimum(dtCFL, minimum(dt_uz, dt_uy) * dtfac)
        # print("dt = ", dtCFL, ", ", dt_uz*0.01, ", ", dt_uy * 0.01, ", ", dt_n * 0.01)
        
        dF_ax2, dF_ay2, dF_bx2, dF_by2, dux2, duy2, duz2, dzz2 = onestep(f, F_ax + dF_ax1/2. * dt, F_ay  + dF_ay1/2. * dt, F_bx + dF_bx1/2. * dt, F_by + dF_by1/2. * dt, ux + dux1/2. * dt, uy + duy1/2. * dt, uz  + duz1/2. * dt, n0, z + dzz1 / 2. * dt, t + dt/2.)
        dF_ax3, dF_ay3, dF_bx3, dF_by3, dux3, duy3, duz3, dzz3 = onestep(f, F_ax + dF_ax2/2. * dt, F_ay  + dF_ay2 /2. * dt, F_bx + dF_bx2 / 2. * dt, F_by + dF_by2 / 2. * dt, ux + dux2 / 2. * dt, uy + duy2 / 2. * dt, uz  + duz2 / 2. * dt, n0, z + dzz2 / 2. * dt, t + dt/2.)    
        dF_ax4, dF_ay4, dF_bx4, dF_by4, dux4, duy4, duz4, dzz4 = onestep(f, F_ax + dF_ax3 * dt, F_ay  + dF_ay3 * dt, F_bx + dF_bx3 * dt, F_by + dF_by3 * dt, ux + dux3 * dt, uy + duy3 * dt, uz  + duz3 * dt, n0, z + dzz3 * dt, t + dt)    
        # time step:
        F_bx += (dF_bx1 + dF_bx2 * 2. + dF_bx3 * 2. + dF_bx4) / 6. * dt
        F_by += (dF_by1 + dF_by2 * 2. + dF_by3 * 2. + dF_by4) / 6. * dt
        F_ax += (dF_ax1 + dF_ax2 * 2. + dF_ax3 * 2. + dF_ax4) / 6. * dt
        F_ay += (dF_ay1 + dF_ay2 * 2. + dF_ay3 * 2. + dF_ay4) / 6. * dt 

        ux += (dux1 + dux2 * 2. + dux3 * 2. + dux4) / 6. * dt
        uy += (duy1 + duy2 * 2. + duy3 * 2. + duy4) / 6. * dt
        uz += (duz1 + duz2 * 2. + duz3 * 2. + duz4) / 6. * dt
        z  += (dzz1 + dzz2 * 2. + dzz3 * 2. + dzz4) / 6. * dt
        # F_n += (dF_n1 + dF_n2 * 2. + dF_n3 * 2. + dF_n4) / 6. * dt
        t += dt

        # velocity damping
        if ifvdamp:
            dampfactor = exp(-dt/dtcay  * exp(-(t-tpack)/dzcay - (z0-z0.min()-tpack - zbuffer)/dzcay)) 
            z = z + (z0-z) * (1.-dampfactor)
            uz *= dampfactor
            uy *= dampfactor
        
        # dyperdiffusion:
        F_bx *= hypercore ;   F_by *= hypercore
        F_ax *= hypercore ;   F_ay *= hypercore  # ;   F_az *= hypercore 
        # F_ux *= hypercore ;   F_uy *= hypercore  ;   F_uz *= hypercore   ;    F_n *= hypercore

        # z cleaning if z(z0) becomes non-monotonic
        #        if (z[1:]-z[:-1]).min() < 0.:
        #            #
        #            uy, uz, n0 = deproject(z, z0, uy, uz, n0) 
        
        if t > (tstore + dtout):
            print("t = ", t)

            if ctr%picture_alias==0:
                # Fourier spectrum:
                plots.fourier(f/2./pi, F_bx, f0, ctr)

            # print(F_bx.real.max(), F_bx.real.min())
            ax = ifft(F_ax).real ;    ay = ifft(F_ay).real  # ;    az = ifft(F_az)
            bx = ifft(F_bx).real ;    by = ifft(F_by).real  # ;    bz = ifft(F_bz)
            # ux = ifft(F_ux) ;    uy = ifft(F_uy)  ;    uz = ifft(F_uz)
            # n = ifft(F_n)
            n = jacoden(z, n0)
            
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

            hio.fewout_dump(hout, ctr, t, (ax, ay), (bx+Bxbgd, by+Bybgd), (ux, uy, uz), n, zcurrent  = z)
            
            # ASCII output
            for k in arange(size(bx)):
                fout.write(str(t) + ' ' + str(z[k]) + ' ' + str(bx[k])+'\n')
            fout.flush()

            mtot = simpson(n.real, x = z)
            epatot = simpson((n*(gamma-1.)).real, x = z)
            emetot = (simpson(bx.real**2+by.real**2, x = z0) + simpson(ax.real**2+ay.real**2, x = z0))/2.

            fout_energy.write(str(t) + ' ' + str(mtot) + ' ' + str(emetot) + ' ' + str(epatot) + '\n')
            fout_energy.flush()
            
            # tlist.append(t)
            mlist.append(mtot)
            paelist.append(epatot)
            emelist.append(emetot)            
            
            if ctr%picture_alias==0:
                # plots.onthefly(z, (z+zlen/2.+t)%zlen-zlen/2., ax0, ay0, az0, bx0 + Bxbgd, by0 + Bybgd, ax, ay, az, bx + Bxbgd, by + Bybgd, ux, uy, uz, n/gamma, ctr, t)
                plots.slew(t, z0, z, ay, bx+Bxbgd, uy, uz, n, ctr)
                
            tlist.append(tstore)
            bxlist.append(bx_prev.real + ((tstore-(t-dt))/dt) * (bx-bx_prev).real + Bxbgd)
            Fbxlist.append(F_bx_prev + ((tstore-(t-dt))/dt) * (F_bx-F_bx_prev))
            # print(len(Fbxlist))
            uylist.append(uy_prev.real +  ((tstore-(t-dt))/dt) * (uy-uy_prev))
            uzlist.append(uz_prev.real +  ((tstore-(t-dt))/dt) * (uz-uz_prev))
            nlist.append((n_prev/gamma_prev).real +  ((tstore-(t-dt))/dt) * (n/gamma - n_prev/gamma_prev))
            tstore += dtout
            ctr += 1

    fout.close()   ; fout_energy.close()
    hout.close()
    
    tlist = asarray(tlist)
    bxlist = asarray(bxlist)
    Fbxlist = asarray(Fbxlist, dtype = complex)
    uylist = asarray(uylist)
    uzlist = asarray(uzlist)
    nlist = asarray(nlist)

    print(shape(uylist))
    
    nt = size(tlist)

    #    print(tlist[1]-tlist[0], dtout)
    print("dtout = ", (tlist[1:]-tlist[:-1]).min(), (tlist[1:]-tlist[:-1]).max())
    
    bxlist_FF = fft(Fbxlist, axis = 0) 
    ofreq = fftfreq(size(tlist), dtout)

    print("omega = ", ofreq)
    print("k = ", f/2./pi)
    
    bxlist_FF = fftshift(bxlist_FF)
    ofreq = fftshift(ofreq)
    f = fftshift(f)
    
    print("omega = ", ofreq)
    print("k = ", f/2./pi)

    # ii = input('T')
    
    # saving the data
    hio.okplane_hout(ofreq*2. * pi, f, bxlist_FF, hname = 'okplane_Bx.hdf', dataname = 'Bx')

    plots.show_nukeplane(omega0 = omega0, bgdfield = Bxbgd)
    
    #    plots.maps(z, tlist, bxlist, uylist, uzlist, nlist, ctr, zalias = 2, talias = 1)

    # final mass and energy plots
    if ifplot:
        plots.maps(z0, tlist, bxlist, uylist, uzlist, nlist, zalias = 4, talias = 1, zcurrent = z)
        
        '''
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
        '''
# plots.show_nukeplane(omega0 = omega0, bgdfield = Bxbgd)
sewerrun()
# ffmpeg -f image2 -r 20 -pattern_type glob -i 'EB*.png' -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  -pix_fmt yuv420p -b 8192k EB.mp4
# uGOcompare('sout_A2_nofeed.hdf5', arange(1000))
