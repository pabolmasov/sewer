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

import matplotlib
from matplotlib.pyplot import *
    
cmap = 'viridis'

from scipy.fft import fft, ifft, fftfreq, fft2
# from scipy.signal import correlate

# simulating a wave moving to the right along z in pair relativistic plasma
# E, B, and v are allowed to have all the three components

# physical switches:
ifmatter = True

# mesh:
nz = 4096
zlen = 25.
z = (arange(nz) / double(nz) - 0.5) * zlen
dz = z[1] - z[0]

# time
# t = 0.
dt = dz * 0.5 # CFL in 1D should be not very small
tmax = 60.
dtout = 0.01

# initial conditions (circularly polirized wave)
z0 = 2.0
f0 = 0.5
amp0 = 0.01
bbgdx = 0.0  ; bbgdy = 0.0 ; bbgdz = 0.0
bx0 = sin(2. * pi * f0 * z) * exp(-(z/z0)**6/2.) * amp0 + bbgdx
by0 =  - cos(2. * pi * f0 * z) * exp(-(z/z0)**6/2. ) * amp0 + bbgdy
bz0 = z *  0. + bbgdz
bz = bz0

ax0 = cos(2. * pi * f0 * z) * exp(-(z/z0)**6/2.) * amp0
ay0 =  sin(2. * pi * f0 * z) * exp(-(z/z0)**6/2.) * amp0
az0 = z * 0.
# 4-velocity
ux0 = 0. * z
uy0 = 0. * z
uz0 = 0. * z
n0 = ones(nz) * 1.0 # density ; let us keep it unity, meaning time is in omega_p units. Lengths are internally in c/f = 2pi c / omega units, that allows a simpler expression for d/dz 

def circorrelate(x, y):

    meanx = x.mean() ; meany = y.mean()
    stdx = x.std() ; stdy = y.std()
    
    nx = size(x)
    r = zeros(nx*2-1)
    
    for k in arange(nx*2-1, dtype = int)-nx:
        y1 = roll(x, k)
        r[k] = ((x-meanx)*(y1-meany)).sum()/stdx/stdy

    return r
        
def onestep(f, F_ax, F_ay, F_az, F_bx, F_by, F_ux, F_uy, F_uz, F_n, ifmatter):
    # one RK4 step
    # avoid interference with the globals!
    
    # essential grid quantities:
    ax = ifft(F_ax) ;    ay = ifft(F_ay) ;    az = ifft(F_az)
    bx = ifft(F_bx) ;    by = ifft(F_by) # ;    bz = ifft(F_bz)
    ux = ifft(F_ux) ;    uy = ifft(F_uy) ;    uz = ifft(F_uz)
    n = ifft(F_n)
    gamma = sqrt(1.+ux**2+uy**2+uz**2)
    vx = ux/gamma ; vy = uy/gamma ; vz = uz/gamma
    
    # Maxwell equations:
    dF_bx = -1.j * f * F_ay # one Maxwell
    dF_by = 1.j * f * F_ax # one Maxwell
    dF_ax = 1.j * f * F_by  - fft(n * ux) * complex(ifmatter) # other Maxwell
    dF_ay = -1.j * f * F_bx - fft(n * uy) * complex(ifmatter) # other Maxwell
    dF_az = - fft(n * uz) * complex(ifmatter) # other Maxwell

    # hydrodynamics
    dF_ux = F_ax + fft( -ifft(-1.j*f*F_ux) + vz * by - vy * by)
    dF_uy = F_ay + fft( -ifft(-1.j*f*F_uy) + vx * bz - vz * bx)
    dF_uz = F_az + fft( -ifft(-1.j*f*F_uz) + vx * by - vy * bx)
    dF_n = 1.j * f * fft(vz*n) - fft(n / gamma**2 * (ax * ux + ay * uy + az * uz ))

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
    hypercore = exp(-(fsq / (fsq.real).max() * (2.*pi)**2)**hyperpower * 3.) + 0.j

    # print(f)
    fout = open('sewerout.dat', 'w+')
    fout.write('# t -- z -- Bx \n')
    
    while t < tmax:
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
            bx = ifft(F_bx) ;    by = ifft(F_by) # ;    bz = ifft(F_bz)
            ux = ifft(F_ux) ;    uy = ifft(F_uy) ;    uz = ifft(F_uz)
            n = ifft(F_n)
            # gamma = sqrt(1.+ux**2+uy**2+uz**2)
            # vx = ux/gamma ; vy = uy/gamma ; vz = uz/gamma
            
            print("Bx = ", bx.min(), '..', bx.max())
            print("Ey = ", ay.min(), '..', ay.max())

            # ASCII output
            for k in arange(size(bx)):
                fout.write(str(t) + ' ' + str(z[k]) + str(bx[k])+'\n')
            fout.flush()
            
            # add a Fourier map
            
            clf()
            fig, axes = subplots(5)
            # print(shape(z), shape(ax))
            axes[0].plot((z+zlen/2.+t)%zlen-zlen/2., ax0, 'r.')
            axes[0].plot(z, ax, 'k-')
            axes[1].plot((z+zlen/2.+t)%zlen-zlen/2., ay0, 'r.')
            axes[1].plot(z, ay, 'k-')
            axes[2].plot((z+zlen/2.+t)%zlen-zlen/2., az0, 'r.')
            axes[2].plot(z, az, 'k-')
            axes[3].plot((z+zlen/2.+t)%zlen-zlen/2., bx0, 'r.')
            axes[3].plot(z, bx, 'k-')
            axes[4].plot((z+zlen/2.+t)%zlen-zlen/2., by0, 'r.')
            axes[4].plot(z, by, 'k-')
            axes[0].set_ylabel(r'$a_x$')
            axes[1].set_ylabel(r'$a_y$')
            axes[2].set_ylabel(r'$a_z$')
            axes[3].set_ylabel(r'$b_x$')
            axes[4].set_ylabel(r'$b_y$')
            axes[4].set_xlabel(r'$z$')  
            # legend()
            axes[0].set_title(r'$t = '+str(round(t))+'$')
            fig.set_size_inches(12.,10.)
            savefig('EB{:05d}.png'.format(ctr))
            clf()
            plot(z, uz, 'k-', label = r'$u^z$')
            plot(z, ux, 'r:', label = r'$u^x$')
            plot(z, uy, 'g--', label = r'$u^y$')
            xlabel(r'$z$')  ;   ylabel(r'$u^i$') 
            legend()
            title(r'$t = '+str(round(t))+'$')
            fig.set_size_inches(12.,5.)
            savefig('u{:05d}.png'.format(ctr))
            clf()
            plot(z, n, 'k-', label = r'$n$')
            xlabel(r'$z$')  ;   ylabel(r'$n$') 
            #        legend()
            title(r'$t = '+str(round(t))+'$')
            fig.set_size_inches(12.,5.)
            savefig('n{:05d}.png'.format(ctr))
            
            close()
            tlist.append(t)
            bxlist.append(bx.real)
            Fbxlist.append(copy(F_bx))
            # print(len(Fbxlist))
            uzlist.append(uz.real)
            nlist.append(n.real)
            tstore += dtout
            ctr += 1

    fout.close()
            
    tlist = asarray(tlist)
    bxlist = asarray(bxlist)
    Fbxlist = asarray(Fbxlist, dtype = complex)
    uzlist = asarray(uzlist).real
    nlist = asarray(nlist).real

    nt = size(tlist)

    # print(Fbxlist.real.max(), Fbxlist.real.min())
    #
    # bxlist_FF = fft2(bxlist) 
    bxlist_FF = fft(Fbxlist, axis = 0) 
    ofreq = fftfreq(size(tlist), dtout)

    # print("wavenumber shape = ", shape(f))
    # print("frequency shape = ", shape(ofreq))
    # print(shape(bxlist_FF))
    # ii = input('T')

    nthalf = nt//2
    nzhalf = nz//2

    # print(ofreq[:nthalf])
    # print(f[:nzhalf])    

    babs = sqrt(bxlist_FF.real**2 + bxlist_FF.imag**2)

    babs = log10(ma.masked_array(babs, mask = (babs <= 0.)))
    
    clf()
    fig = figure()
    # pcolormesh(tlist, f/2./pi, transpose(Fbxlist.real), shading='nearest', vmin = -10, vmax = 10)
    # pcolormesh(ofreq, f/2./pi, transpose((bxlist_FF.real**2 + bxlist_FF.imag**2)))
    pcolormesh(ofreq[:(nthalf+1)], f[:nzhalf]/2./pi, transpose(babs)[:nzhalf, :nthalf], vmin = maximum(babs.min(), babs.max()-5.), vmax = babs.max(), shading='nearest')
    pcolormesh(ofreq[nthalf:], f[:nzhalf]/2./pi, transpose(babs)[:nzhalf, nthalf:], vmin = maximum(babs.min(), babs.max()-5.), vmax = babs.max(), shading='nearest')
    pcolormesh(ofreq[:nthalf], f[nzhalf:]/2./pi, transpose(babs)[nzhalf:, :nthalf], vmin = maximum(babs.min(), babs.max()-5.), vmax = babs.max(), shading='nearest')
    pcolormesh(ofreq[nthalf:], f[nzhalf:]/2./pi, transpose(babs)[nzhalf:, nthalf:], vmin = maximum(babs.min(), babs.max()-5.), vmax = babs.max(), shading='nearest')
    cb = colorbar()
    cb.set_label(r'$\log_{10} |\tilde{b}_x|$')
    plot([-f0], [ f0], 'ro', mfc='none')
    plot([f0], [ -f0], 'ro', mfc='none')

    ktmp = 15.*f0 * (arange(100)/double(100)-0.5)

    plot(ktmp/2./pi, ktmp/2./pi, 'w--')
    plot(-ktmp/2./pi, ktmp/2./pi, 'w--')
    plot(ktmp/2./pi, -ktmp/2./pi, 'w--')
    plot(-ktmp/2./pi, -ktmp/2./pi, 'w--')    
    
    if abs(bbgd)>0.01:
        # circularly polarized components
        plot(ktmp/2./pi, (ktmp - 1./(ktmp+bbgd))/2./pi, 'b:')
        plot(ktmp/2./pi, (ktmp - 1./(ktmp-bbgd))/2./pi, 'r:')        
        plot(ktmp/2./pi, -(ktmp - 1./(ktmp+bbgd))/2./pi, 'r:')
        plot(ktmp/2./pi, -(ktmp - 1./(ktmp-bbgd))/2./pi, 'b:')        
    else:
        plot(sqrt(1.+ktmp**2)/2./pi, ktmp/2./pi, 'w:')
        plot(-sqrt(1.+ktmp**2)/2./pi, ktmp/2./pi, 'w:')
        plot(sqrt(1.+ktmp**2)/2./pi, -ktmp/2./pi, 'w:')
        plot(-sqrt(1.+ktmp**2)/2./pi, -ktmp/2./pi, 'w:')

        
        
    xlim(-2. * f0 , 2. * f0 )
    ylim(-2. * f0, 2. * f0)
    
    #    xlim(1./tmax, 1./dtout)  ;  ylim(1./zlen, 1./dz)
    fig.set_size_inches(15.,10.)
    xlabel(r'$\omega$') ; ylabel(r'$k$')    
    savefig('okplane.png')
    
    
    # cross-correlations
    # print(shape(bx0), shape(bx))
    #    ccor_bx = correlate(bx0, bx0, mode='full', method='fft')
    # ddz = -(arange(2*nz-1)-double(nz))*dz
    # print(shape(ccor_bx), shape(ddz))
    # clf()
    #for k in arange(nt//5)*5:
    #     ccor_bx = circorrelate(bx0, bx0) #, mode='full') # , method='fft')         
    #     plot(ddz-tlist[k] * 0., ccor_bx+tlist[k] * 0.1, 'k.')
         # cmax = ccor_bx[(ddz > (tlist[k]-2.*pi/f0))&(ddz > (tlist[k]+2.*pi/f0))].max()
         # plot([0., 0.], [cmax+tlist[k] * 0.1, cmax+tlist[k] * 0.1], 'rx')

    # xlim( - dz * 100., dz * 100.)
    # xlabel(r'$\Delta z$') ; ylabel(r'$C$')  
    # savefig('ccor.png')

    #     print(bxlist)
     
    # 2D-visualization
    clf()
    pcolormesh(z, tlist, bxlist)
    colorbar()
    plot(z, z, 'w--')
    ylim(tlist.min(), tlist.max())
    xlabel(r'$z$') ; ylabel(r'$t$')  
    fig.set_size_inches(15.,10.)
    savefig('EBmap.png'.format(ctr))
    
    clf()
    pcolormesh(z, tlist, uzlist)
    colorbar()
    xlabel(r'$z$') ; ylabel(r'$t$')  
    fig.set_size_inches(15.,10.)
    savefig('uzmap.png'.format(ctr))
    
    clf()
    pcolormesh(z, tlist, nlist)
    colorbar()
    xlabel(r'$z$') ; ylabel(r'$t$')  
    fig.set_size_inches(15.,10.)
    savefig('nmap.png'.format(ctr))
    
    close()
        

# ffmpeg -f image2 -r 20 -pattern_type glob -i 'EB*.png' -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  -pix_fmt yuv420p -b 8192k EB.mp4
