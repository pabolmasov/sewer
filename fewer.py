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

# from scipy.fft import fft, ifft, fftfreq, fft2, fftshift
# from scipy.signal import correlate
from scipy.interpolate import make_interp_spline
from scipy.integrate import simpson

# HDF5 io:
import hio

# plotting
import plots

# simulating a wave moving to the right along z in pair relativistic plasma
# E, B, and v are allowed to have all the three components

# physical switches:
ifmatter = False # feedback
ifuz = False
ifnowave = False # no EM wave, but the initial velocities are perturbed

rtol = 1e-8

nmin = 0.0

nghost = 2
nspline = nghost+1

ndigits = 2

# TODO: energy check

# mesh:
nz = 1024
zlen = 40.
zbuffer = 2.0
z = (arange(nz) / double(nz) - 0.5) * zlen
z_ext = ((arange(nz+ 2 * nghost) -double(nghost))/ double(nz) - 0.5) * zlen
zhalf = (z[1:]+z[:-1])/2. # edges
dz = z[1] - z[0]
print(z_ext.min(), z_ext.max())
print("dz = ", dz)

# time
dtCFL = dz * 0.1 # CFL in 1D should be not very small
dtfac = 0.01
dtout = 0.01
picture_alias = 100

# injection:
ExA = 0.0
EyA = 100.0
omega0 = 20.0
tpack = sqrt(2.5)
tmid = tpack * 10. # the passage of the wave through z=0
tmax = 3. * tmid

if ifnowave:
    EyA = 0.
    ExA = 0.

Bz = 0.0
Bxbgd = 0.0
Bybgd = 0.0

def Aleft(t):
    return -sin(omega0 * (t -tmid + dz/2.)) * exp(-((t + dz/2.-tmid)/tpack)**2/2.) / omega0

def Eleft(t):
    return (cos(omega0 * (t-tmid)) - (t-tmid)/(omega0*tpack**2) * sin(omega0 * (t-tmid))) * exp(-((t-tmid)/tpack)**2/2.)

def Bleft(t):
    return Eleft(t + dz/2.)
# sin(omega0 * (t-tmid+dz/2.)) * exp(-((t+dz/2.-tmid)/tpack)**2/2.)

def phiRL(uside, v):
    # slope limiter
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

def parcoeff(v):
     
    acoeff = (v[2:]+v[:-2]-2. * v[1:-1])/2.
    bcoeff = (v[2:]-v[:-2])/2.
    ccoeff = v[1:-1]

    return acoeff, bcoeff, ccoeff

def phifun(r1, r2):
    w0 = (r1**2+r2**2) < rtol
    w1 = (r1**2+r2**2) >= rtol
    r = copy(r1)

    if w1.sum() > 0:
        r[w1] = (2. * r1 * r2**2 / (r1**2+r2**2))[w1] # van Albada
        # ((r1*sign(r2)+abs(r1))/(abs(r2)+abs(r1)))[w1] # van Leer's, nominator and denominator * abs(r2)
    
    if w0.sum() > 0:
        r[w0] = 1.
    
    #    print(r.min(), r.max())

    return r

def dvstep_parabolic(ux, uy, uz, n, Ex, Ey, Bx, By):

    gamma = sqrt(1. + ux**2 + uy**2 + uz**2)

    uy_acoeff, uy_bcoeff, uy_ccoeff = parcoeff(uy)
    ux_acoeff, ux_bcoeff, ux_ccoeff = parcoeff(ux)
    uz_acoeff, uz_bcoeff, uz_ccoeff = parcoeff(uz)

    vx = ux / gamma ; vy = uy / gamma ; vz =  uz / gamma

    vz_acoeff, vz_bcoeff, vz_ccoeff = parcoeff(vz)
    n_acoeff, n_bcoeff, n_ccoeff = parcoeff(maximum(n, 0.))

    # mean non-linear terms:
    dux = (ux_bcoeff * vz_acoeff + 2. * ux_acoeff * vz_bcoeff) / 12. + ux_bcoeff * vz_ccoeff
    duy = (uy_bcoeff * vz_acoeff + 2. * uy_acoeff * vz_bcoeff) / 12. + uy_bcoeff * vz_ccoeff
    duz = (uz_bcoeff * vz_acoeff + 2. * uz_acoeff * vz_bcoeff) / 12. + uz_bcoeff * vz_ccoeff
    dux /= dz ; duy /= dz  ; duz /= dz

    dux += (Ex[1:]+Ex[:-1])/2.
    duy += (Ey[1:]+Ey[:-1])/2.
    
    dux += (vy * Bz - vz * By)[1:-1]
    duy += (vz * Bx - vx * Bz)[1:-1]
    duz += (vx * By - vy * Bx)[1:-1]

    # mass flux
    # nz = ((n * vz)[1:] - (n * vz)[:-1])/dz

    wrightwind = (vz[1:-1] >= 0.)
    wleftwind = (vz[1:-1] < 0.)
    
    dn = zeros(nz)
    
    # zshift = zhalf - vz * dt
    
    dnleft = (vz_acoeff / 4. - vz_bcoeff / 2. + vz_ccoeff) * (n_bcoeff - n_acoeff) # vz dn/dz halfcell left 
    dnright = (vz_acoeff / 4. + vz_bcoeff / 2. + vz_ccoeff) * (n_bcoeff + n_acoeff) # vz dn/dz halfcell right 
    dnleft += (n_acoeff / 4. - n_bcoeff / 2. + n_ccoeff) * (vz_bcoeff - vz_acoeff) 
    dnright += (n_acoeff / 4. + n_bcoeff / 2. + n_ccoeff) * (vz_bcoeff + vz_acoeff)

    if wrightwind.sum() > 0:
        dn[wrightwind] =  dnleft[wrightwind]
    if wleftwind.sum() > 0:
        dn[wleftwind] = dnright[wleftwind]

    # dn += vz_bcoeff * n_ccoeff        
    dn = (n_bcoeff * vz_acoeff + 2. * n_acoeff * vz_bcoeff) / 12. + n_bcoeff * vz_ccoeff + (vz_bcoeff * n_acoeff + 2. * vz_acoeff * n_bcoeff) / 12. + vz_bcoeff * n_ccoeff

    dn = -dn/dz
    # post-factum dt estimate

    dt_post_n = abs(n[1:-1] / dn)[n[1:-1]>0.5].min()
    dt_post_v = 1./maximum(abs(duy), abs(duz)).max()
    
    return (dux, duy, duz), dn, minimum(dt_post_n, dt_post_v)
    
def dvstep(ux, uy, uz, n, Ex, Ey, Bx, By):
    # upwind integration

    gamma = sqrt(1. + ux**2 + uy**2 + uz**2)

    vx = ux / gamma ; vy = uy / gamma ; vz = uz / gamma

    vzhalf = (vz[1:]+vz[:-1])/2.
    
    dux_side = - vzhalf * (ux[1:]-ux[:-1])/dz # nz+1
    duy_side = - vzhalf * (uy[1:]-uy[:-1])/dz
    duz_side = - vzhalf * (uz[1:]-uz[:-1])/dz
    dn_side = - ((n * vz)[1:] - (n*vz)[:-1])/dz
    
    dux = (vy * Bz - vz * By)[1:-1]
    duy = (vz * Bx - vx * Bz)[1:-1]
    duz = (vx * By - vy * Bx)[1:-1]
    
    dux += phiRL(dux_side + Ex, vzhalf) 
    duy += phiRL(duy_side + Ey, vzhalf) 
    duz += phiRL(duz_side, vzhalf)
    dn = phiRL(dn_side, vzhalf)
    
    return (dux, duy, duz), dn 

def dsteps(t, E, B, u, n):

    Ex, Ey = E
    Bx, By = B
    ux, uy, uz = u

    # boundary values
    if ifuz:
        uz0 = (ExA**2 + EyA**2) * Aleft(t)**2/2.
        uy0 = -EyA * Aleft(t) - Bxbgd * dz
        ux0 = ExA * Aleft(t) + Bybgd * dz
        n0 = sqrt(1.+uy0**2+ux0**2+uz0**2)
    else:
        # n0 = n[0]
        uz0 = uz[0] # minimum(uz[0], 0.)
        uy0 = uy[0] # - (Bxbgd-EyA*Eleft(t) * 0.) * dz
        ux0 = ux[0] # + (Bybgd+ExA*Eleft(t) * 0.) * dz
        # uz0 = uz[0] + (uy[0]+uy0) * (Bxbgd-EyA*Eleft(t)) * dz

    if uz[0] >= 0.:
        n0 = sqrt(1.+uy0**2+ux0**2+uz0**2)
    else:
        n0 = n[0]

    n0 = n[0]

    # n0 = sqrt(1.+uy0**2+ux0**2+uz0**2)
    # n0 = 1. ; n[0] = (sqrt(1.+uy**2+ux**2+uz**2))[0]
    # n0 = (n*sqrt(1.+uy**2+ux**2+uz**2))[0]
        
    if uz[-1] >= 0.:
        n1 = n[-1] 
    else:
        n1 = sqrt(1.+uy**2+ux**2+uz**2)[-1]
        
    n1 = n[-1]

    # extended arrays
            
    #Ex_ext1 = concatenate([[ExA * Eleft(t)]] +  [Ex]  + [[(By[-1] - Bybgd + Ex[-1])/2.]])
    #Ey_ext1 = concatenate([[EyA * Eleft(t)]] + [Ey] +  [[(-Bx[-1] + Bxbgd +Ey[-1])/2.]])
    Ex_ext = concatenate([[ExA * Eleft(t)]] +  [Ex]  + [[By[-1]]]) # - Bybgd + Ex[-1])/2.]] )
    Ey_ext = concatenate([[EyA * Eleft(t)]]  + [Ey] +  [[Bxbgd-Bx[-1]]]) # + Bxbgd +Ey[-1])/2.]] )
    Bx_ext = concatenate([[EyA * Bleft(t)+Bxbgd]]  +  [Bx] + [[Bx[-1]]] )
    By_ext = concatenate([[-ExA * Bleft(t)+Bybgd]] +  [By] +  [[By[-1]]])
    ux_ext = concatenate([[ux0]] + [ux] +  [[ux[-1]]] )
    uy_ext = concatenate([[uy0]] +  [uy] + [[uy[-1]]])
    uz_ext = concatenate([[uz0]] +  [uz] + [[uz[-1]]])
    n_ext = concatenate([[n0]] +  [n] +  [[n1]])

    # print(size(Bx_ext))
    # print(n_ext)
    # ii = input('N')
    
    # currents
    gamma = sqrt(1. + ux**2 + uy**2 + uz**2 )
    if ifmatter:
        # gamma = sqrt(1. + ux**2 + uy**2 + uz**2 )
        jx = -n * ux/gamma ;     jy = -n * uy/gamma ;         jz = 0.
    else:
        jx = 0. ; jy = 0. ; jz = 0.
    
    dB = dBstep(Ex_ext, Ey_ext)
    dE = dEstep(Bx, By, jx, jy, uz/gamma)
    du, dn, dt_post = dvstep_parabolic(ux_ext, uy_ext, uz_ext, n_ext, Ex_ext, Ey_ext, Bx_ext, By_ext)
    
    return dE, dB, du, dn, dt_post
        
def sewerrun():
    
    # E on the edges, B in cell centres (Bz is not evolves, just postulated)
    Bx = -EyA * Bleft(z.min()-dz-z) # z.min()-dz is the coord of the ghost zone
    By = ExA * Bleft(z.min()-dz -z)

    Bx += Bxbgd
    By += Bybgd
    
    Ex = ExA * Eleft(z.min()-dz-zhalf) # ghost zone + dz/2.
    Ey = EyA * Eleft(z.min()-dz-zhalf)
    #     Ex = zeros(nz)

    ux = zeros(nz)  ; uy = zeros(nz) ; uz = zeros(nz) ; n = ones(nz)

    if ifnowave:
        uy += exp(-0.5*(z/tpack)**2) * 0.1
        uylist = [] ; uzlist = []
        
    n *= (z > (z.min()+zbuffer)) & (z < (z.max()-zbuffer)) # sin(z * 10.) * 0.1

    # total quantities:
    tlist = []
    mlist = [] # particle mass
    paelist = [] # particle energy
    emelist = [] # EM fields energy
    
    t = 0. ; ctr = 0; figctr = 0
    
    while ((t < tmax) & (abs(Bx).max() < ( 1e6/omega0)) & (abs(uy).max() < ( 1e6/omega0)) & isfinite(n.max())):

        # first step in RK4
        dE, dB, du, dn1, dt1 = dsteps(t, (Ex, Ey), (Bx, By), (ux, uy, uz), n)
        dEx1, dEy1 = dE    ;   dBx1, dBy1 = dB   ; dux1, duy1, duz1 = du

        # adaptive time step:
        dt = minimum(dtCFL, dt1 * dtfac)
        #if (dt < dtCFL):
        #    print("dt = ", dt)
        
        # second step in RK4
        dE, dB, du, dn2, dt2 = dsteps(t+dt/2., (Ex+dEx1 * dt/2., Ey+dEy1 * dt/2.),
                            (Bx+dBx1*dt/2., By+dBy1*dt/2.), (ux+dux1*dt/2., uy+duy1*dt/2., uz+duz1*dt/2.), n+dn1*dt/2.)
        dEx2, dEy2 = dE    ;   dBx2, dBy2 = dB   ; dux2, duy2, duz2 = du
        
        # third step in RK4
        dE, dB, du, dn3, dt3 = dsteps(t+dt/2., (Ex+dEx2 * dt/2., Ey+dEy2 * dt/2.),
                            (Bx+dBx2*dt/2., By+dBy2*dt/2.), (ux+dux2*dt/2., uy+duy2*dt/2., uz+duz2*dt/2.), n+dn2*dt/2.)
        dEx3, dEy3 = dE    ;   dBx3, dBy3 = dB   ; dux3, duy3, duz3 = du

        # fourth step
        dE, dB, du, dn4, dt4 = dsteps(t+dt, (Ex+dEx3 * dt, Ey+dEy3 * dt), (Bx+dBx3*dt, By+dBy3*dt),
                                 (ux+dux3*dt, uy+duy3*dt, uz+duz3*dt), n+dn3*dt)
        dEx4, dEy4 = dE    ;   dBx4, dBy4 = dB   ; dux4, duy4, duz4 = du

        # the actual time step
        Bx = Bx + (dBx1 + 2. * dBx2 + 2. * dBx3 + dBx4) * dt/6. ; By = By + (dBy1 + 2. * dBy2 + 2. * dBy3 + dBy4) * dt/6.
        Ex = Ex + (dEx1 + 2. * dEx2 + 2. * dEx3 + dEx4) * dt/6. ; Ey = Ey + (dEy1 + 2. * dEy2 + 2. * dEy3 + dEy4) * dt/6.
        ux = ux + (dux1 + 2. * dux2 + 2. * dux3 + dux4) * dt/6. ; uy = uy + (duy1 + 2. * duy2 + 2. * duy3 + duy4) * dt/6.
        uz = uz + (duz1 + 2. * duz2 + 2. * duz3 + duz4) * dt/6. ; n = n + (dn1 + 2. * dn2 + 2. * dn3 + dn4) * dt/6.

        n = maximum(n, nmin)
        
        t += dt ; ctr += 1
        if ctr%picture_alias==0:

            # totals:
            gamma = sqrt(1. + ux**2 + uy**2 + uz**2 )

            mtot = simpson(n, x = z)
            epatot = simpson(n*(gamma-1.), x = z)
            emetot = (simpson(Bx**2+By**2, x = z) + simpson(Ex**2+Ey**2, x = zhalf))/2.
            
            tlist.append(t)
            mlist.append(mtot)
            paelist.append(epatot)
            emelist.append(emetot)

            if ifnowave:
                uylist.append(uy[nz//2])
                uzlist.append(uz[nz//2])
            
            print("mlist = ", simpson(n/gamma, x = z))
            print("E_PA + E_EM <= ", epatot, " + ", emetot, " = ", epatot + emetot)
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

            uytmp = arange(100)/100. * (uy.max() - uy.min()) + uy.min()
            clf()
            plot(uytmp, uytmp**2/2., 'r-')
            scatter(uy, uz, c = z)
            cb = colorbar()
            cb.set_label(r'$z$')
            xlabel(r'$u^y$')   ;   ylabel(r'$u^z$')
            # xlim(-15, -10) ; ylim(-0.1,0.1)
            title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
            savefig('GO{:05d}.png'.format(figctr))
            clf()
            plot(z, 1.+(Aleft(t-z+z.min())*EyA)**2/2., 'r:')
            plot(z[n>0.], n[n>0.], '-k')
            # cb = colorbar()
            # cb.set_label(r'$z$')
            xlabel(r'$z$')   ;   ylabel(r'$n_{\rm p}$') 
            title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
            savefig('n{:05d}.png'.format(figctr))
            close()
            
            # HDF5 dump:
            if figctr == 0:
                hout = hio.fewout_init('fout.hdf5',
                                       {"ifmatter": ifmatter, "ExA": ExA, "EyA": EyA,
                                        "omega0": omega0, "tpack": tpack, "tmid": tmid, "Bz": Bz, "Bx": Bxbgd},
                                       z, zhalf = zhalf)

            hio.fewout_dump(hout, figctr, t, (Ex, Ey), (Bx, By), (ux, uy, uz), n)            
            print("dt = ", dt)
            ctr = 0 ; figctr += 1

    hout.close()

    paelist  = asarray(paelist)  ; emelist = asarray(emelist)  ; tlist = asarray(tlist)
    
    # final mass and energy plots
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
    if ifnowave:
        clf()
        plot(tlist, 0.1 * cos(Bxbgd*tlist), 'g:')
        plot(tlist, -0.1 * sin(Bxbgd*tlist), 'b-.')
        plot(tlist, uylist, 'k.', label = r'$u^y$')
        plot(tlist, uzlist, 'rx', label = r'$u^z$')
        legend()
        xlabel(r'$t$')  ;  ylabel(r'$u^{y, z}(z=0)$')
        savefig('uosc.png')
        clf()
        plot(0.1 * cos(Bxbgd*tlist), -0.1 * sin(Bxbgd*tlist), 'k-')
        scatter(uylist, uzlist, c = tlist*Bxbgd)
        cb = colorbar()
        cb.set_label(r'$\omega_{\rm c} t$')
        xlabel(r'$u^y$')  ;  ylabel(r'$u^{z}$')
        savefig('uosc_circle.png')
