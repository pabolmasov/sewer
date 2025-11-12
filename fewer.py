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
from scipy.fft import fft, ifft, fftfreq, fft2, fftshift

# parallel support
from mpi4py import MPI
# MPI parameters:
comm = MPI.COMM_WORLD
crank = comm.Get_rank()
csize = comm.Get_size()
first = 0 ; last = csize-1
left = crank-1 ; right = crank+1

# HDF5 io:
import hio

# plotting
import plots

# various auxiliary routines
import utile

# simulating a wave moving to the right along z in pair relativistic plasma
# E, B, and v are allowed to have all the three components

# physical switches:
ifmatter = False # feedback
ifuz = True # left BC for the velocities; without MF, True works much better
ifnowave = False # no EM wave, but the initial velocities are perturbed
ifpar = False
ifBbuffer = True
ifvdamp = True
ifexpdamp  = True

rtol = 1e-8
vmin = 1e-8

nmin = 0.0

nghost = 2
nspline = nghost+1

ndigits = 2 # for output, t is truncated to ndigits after .

# TODO: energy check

# mesh:
nz = 2048  # per core
nc = csize # number of cores
zlen = 60.
zbuffer = 5.0
z = (arange(nz*nc) / double(nz*nc) - 0.5) * zlen
dz = z[1] - z[0]
if csize > 1:
    z = array_split(z, csize)[crank]

zhalf = z - dz/2. # (z[1:]+z[:-1])/2. # edges
z_ext = concatenate([[z[0]-dz], z, [z[-1]+dz]])

print(z_ext.min(), z_ext.max())
print("dz = ", dz)

print("core ", crank, ": z_ext = ", z_ext[0], "..", z_ext[-1])

# time
dtCFL = dz * 0.1 # CFL in 1D should be not very small
dtfac = 0.1
dtcay = 3.
dzcay = 3.

# dtout = 0.01
ifplot = True
hdf_alias = 100
picture_alias = 100

# injection:
ExA = 0.0
EyA = 80.0
omega0 = 40.0
tpack = sqrt(6.)
tmid = tpack * 10. # the passage of the wave through z=0
tmax = zlen + tmid

tstart = 0.

print("tmax = ", tmax)

if ifnowave:
    EyA = 0.
    ExA = 0.

Bz = 0.0
Bxbgd = 5.0
Bybgd = 0.0

# Fourier smoothing
f = fftfreq(nz, d = dz / (2. * pi)) # Fourier mesh
fsq = (f * conj(f)).real

def sign(x):
    if x > 0.:
        return 1.
    elif x < 0.:
        return -1.
    return 0.

def Aleft(t):
    return -sin(omega0 * (t -tmid - dz/2.)) * exp(-((t - dz/2.-tmid)/tpack)**2/2.) / omega0

def Eleft(t):
    return (cos(omega0 * (t-tmid)) - (t-tmid)/(omega0*tpack**2) * sin(omega0 * (t-tmid))) * exp(-((t-tmid)/tpack)**2/2.)

def Bleft(t):
    return Eleft(t - dz/2.)
# sin(omega0 * (t-tmid+dz/2.)) * exp(-((t+dz/2.-tmid)/tpack)**2/2.)

        
def buffermod(zz):
    znorm = -(zlen/2. - zbuffer)
    dzstep = zbuffer
    bfr = utile.bufferfun((zz-znorm)/dzstep)
    
    return bfr # (zz > -znorm) * (zz < znorm)   * (zz / znorm + 1.) * (1. - zz/znorm) 

def Bbuffermod(zz):
    if ifBbuffer:
        return buffermod(zz)
    else:
        return zz*0.+1.

def dBstep(Ex, Ey):
    dBx = zeros(nz) ;  dBy = zeros(nz) # ; v = ones(nz+1)

    #using extended E arrays with the BCs
    
    dBx = (Ey[2:]-Ey[1:-1]) / dz
    dBy = -(Ex[2:]-Ex[1:-1]) / dz
    
    return dBx, dBy

def dEstep(Bx, By, jx, jy, v):

    dEx = zeros(nz)
    dEy = zeros(nz)

    #    print("jx = ", size(jx), "; v = ", size(v))

    dEx = - (By[1:-1]-By[:-2])/dz # Bx, By, and j should be extended
    dEy = (Bx[1:-1]-Bx[:-2])/dz
    
    if ifmatter:
        dEx += utile.phiRL(jx, v)[1:]
        dEy += utile.phiRL(jy, v)[1:]
    
    return dEx, dEy

def dvstep_parabolic(ux, uy, uz, n, s, Ex, Ey, Bx, By):

    # uz = maximum(uz, 0.)
    
    gamma = sqrt(1. + ux**2 + uy**2 + uz**2)

    uy_acoeff, uy_bcoeff, uy_ccoeff = utile.parcoeff(uy)
    ux_acoeff, ux_bcoeff, ux_ccoeff = utile.parcoeff(ux)
    uz_acoeff, uz_bcoeff, uz_ccoeff = utile.parcoeff(uz)

    vx = ux / gamma ; vy = uy / gamma ; vz =  uz / gamma
    
    vz_acoeff, vz_bcoeff, vz_ccoeff = utile.parcoeff(vz)
    n_acoeff, n_bcoeff, n_ccoeff = utile.parcoeff(maximum(n, 0.))
    s_acoeff, s_bcoeff, s_ccoeff = utile.parcoeff(s)

    # mean non-linear terms:    
    
    dux = (ux_bcoeff * vz_acoeff + 2. * ux_acoeff * vz_bcoeff) / 12. + ux_bcoeff * vz_ccoeff
    duy = (uy_bcoeff * vz_acoeff + 2. * uy_acoeff * vz_bcoeff) / 12. + uy_bcoeff * vz_ccoeff
    duz = (uz_bcoeff * vz_acoeff + 2. * uz_acoeff * vz_bcoeff) / 12. + uz_bcoeff * vz_ccoeff
    
    ''' # parabolic (above) has much better accuracy
    dux = 2. * vz_ccoeff * ux_bcoeff 
    duy = 2. * vz_ccoeff * uy_bcoeff
    duz = 2. * vz_ccoeff * uz_bcoeff 
    '''    
   # what if vz_max > 1? then
    '''
    vzpeak = vz_ccoeff - vz_bcoeff**2/4./vz_acoeff
    zpeak = - vz_bcoeff / 2./ vz_acoeff
    wvz = (abs(vzpeak)>1.) & (zpeak <= 1.) & (zpeak >= -1.)

    if wvz.sum() >= 1.:
        print(wvz.sum(), "points with vzpeak >= 1")
        print("extreme vz = ", vzpeak.min(), '..', vzpeak.max(), "\n\n")
        print(vz_ccoeff.min(), vz_ccoeff.max())
        # ii = input("I")
        dux[wvz] /= abs(vzpeak)[wvz]
        duy[wvz] /= abs(vzpeak)[wvz]
        duz[wvz] /= abs(vzpeak)[wvz]
    '''
    dux /= -dz ; duy /= -dz  ; duz /= -dz
    
    dux += (Ex[2:]+Ex[1:-1])/2.
    duy += (Ey[2:]+Ey[1:-1])/2.
    
    dux += (vy * Bz - vz * (By+Bybgd * buffermod(z_ext)))[1:-1]
    duy += (vz * (Bx+Bxbgd * Bbuffermod(z_ext)) - vx * Bz)[1:-1]
    duz += (vx * (By+Bybgd * Bbuffermod(z_ext)) - vy * (Bx+Bxbgd * Bbuffermod(z_ext)))[1:-1]

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

    ds = (s_bcoeff * vz_acoeff + 2. * s_acoeff * vz_bcoeff) / 12. + s_bcoeff * vz_ccoeff
    ds = -ds/dz
    
    # post-factum dt estimate

    wn = n[1:-1]>0.1
    if wn.sum() > 0:
        dt_post_n = abs(n[1:-1] / dn)[wn].min()
    else:
        # print("max(n) = ", n[1:-1].max())
        dt_post_n = tmax # in this case, dn is irrelevant
  
    # if ifvdamp:
    #    duD = exp(minimum( 5. - (z-z.min()) * 0.5,1.)) # / (dtcay * dtCFL)
        # print("duD = ", duD.min(), ".. ", duD.max())
        # duy -= duD * uy[1:-1]
    #    duz -= duD * uz[1:-1]        

    wyz = (abs(uy[1:-1]) > vmin) & (abs(uz[1:-1]) > vmin) & (abs(duy) > vmin) & (abs(duz) > vmin)     

    if wyz.sum() > 1:
        dt_post_v = 1./maximum(abs(duy[wyz]), abs(duz[wyz])).max() #!!! removes [wyz]
    else:
        dt_post_v = tmax
    return (dux, duy, duz), dn, ds, minimum(dt_post_n, dt_post_v)
    
def dsteps(t, E, B, u, n, s):

    Ex, Ey = E
    Bx, By = B
    ux, uy, uz = u

    # left BC:
    if crank == first:
        Ex0 = ExA * Eleft(t)
        Ey0 = EyA * Eleft(t)
        Bx0 = EyA * Bleft(t)
        By0 = -ExA * Bleft(t)
        if ifuz:
            uz0 = (ExA**2 + EyA**2) * Aleft(t)**2/2.
            uy0 = EyA * Aleft(t) # - Bxbgd * dz
            ux0 = ExA * Aleft(t) # + Bybgd * dz
            n0 = 0. # sqrt(1.+uy0**2+ux0**2+uz0**2)
        else:
            uz0 =  sqrt(maximum(uz[0]**2 + 2. * uy[0] * Bxbgd*dz, 0.)) # uz[0] - (uz[1]-uz[0])
            # print(uz0)
            uy0 = uy[0]  - Bxbgd * dz
            ux0 = ux[0]  + Bybgd * dz # - (ux[1]-ux[0])# + (Bybgd+ExA*Eleft(t) * 0.) * dz
            if ifpar:
                 uz0_acoeff, uz0_bcoeff, uz0_ccoeff = utile.parcoeff(uz[0:3])
                 uy0_acoeff, uy0_bcoeff, uy0_ccoeff = utile.parcoeff(uy[0:3])
                 uz0 = 4. * uz0_acoeff[0] - 2. * uz0_bcoeff[0] + uz0_ccoeff[0]
                 uy0 = 4. * uy0_acoeff[0] - 2. * uy0_bcoeff[0] + uy0_ccoeff[0]
            n0 = n[0]
        s0 = -zlen/2. # sqrt(1.+uy0**2+ux0**2+uz0**2)
    else:
        leftpack = {"Ex": Ex[0], "Ey": Ey[0], "Bx": Bx[0], "By": By[0],
                    "ux": ux[0], "uy": uy[0], "uz": uz[0], "n": n[0], "s": s[0]}
        comm.send(leftpack, dest = left, tag = crank) # sending all the boundary values to the left

    if crank == last:
        Ex1 = By[-1]
        Ey1 = - Bx[-1]
        Bx1 = Bx[-1]
        By1 = By[-1]
        uz1 = maximum(uz[-1], 0.)

        uy1 = uy[-1]
        ux1 = ux[-1]

        n1 = n[-1]
        s1 = zlen/2.
    else:
        rightpack = {"Ex": Ex[-1], "Ey": Ey[-1], "Bx": Bx[-1], "By": By[-1],
                     "ux": ux[-1], "uy": uy[-1], "uz": uz[-1], "n": n[-1], "s": s[-1]}
        comm.send(rightpack, dest = right, tag = crank) # sending all the boundary values to the right

    # receiving data from the neighbouring blocks
    if crank < last:
        rightpack = comm.recv(source = right, tag = right)
        Ex1 = rightpack["Ex"] ; Ey1 = rightpack["Ey"] 
        Bx1 = rightpack["Bx"] ; By1 = rightpack["By"] 
        ux1 = rightpack["ux"] ; uy1 = rightpack["uy"]  ; uz1 = rightpack["uz"] ; n1 = rightpack["n"] ; s1 = rightpack["s"]
    if crank > first:
        leftpack = comm.recv(source = left, tag = left)
        Ex0 = leftpack["Ex"] ; Ey0 = leftpack["Ey"] 
        Bx0 = leftpack["Bx"] ; By0 = leftpack["By"] 
        ux0 = leftpack["ux"] ; uy0 = leftpack["uy"]  ; uz0 = leftpack["uz"] ; n0 = leftpack["n"] ; s0 = leftpack["s"]
       
    # extended arrays
            
    #Ex_ext1 = concatenate([[ExA * Eleft(t)]] +  [Ex]  + [[(By[-1] - Bybgd + Ex[-1])/2.]])
    #Ey_ext1 = concatenate([[EyA * Eleft(t)]] + [Ey] +  [[(-Bx[-1] + Bxbgd +Ey[-1])/2.]])
    Ex_ext = concatenate([[Ex0]] + [Ex] + [[Ex1]]) # - Bybgd + Ex[-1])/2.]] )
    Ey_ext = concatenate([[Ey0]] + [Ey] +  [[Ey1]]) # + Bxbgd +Ey[-1])/2.]] )
    Bx_ext = concatenate([[Bx0]] + [Bx] + [[Bx1]] )
    By_ext = concatenate([[By0]] + [By] +  [[By1]])
    ux_ext = concatenate([[ux0]] + [ux] +  [[ux1]] )
    uy_ext = concatenate([[uy0]] + [uy] + [[uy1]])
    uz_ext = concatenate([[uz0]] + [uz] + [[uz1]])
    n_ext = concatenate([[n0]] +  [n] +  [[n1]])
    s_ext = concatenate([[s0]] +  [s] +  [[s1]])

    # print(size(Bx_ext))
    # print(n_ext)
    # ii = input('N')
    
    # currents
    gamma = sqrt(1. + ux_ext**2 + uy_ext**2 + uz_ext**2 )
    if ifmatter:
        # gamma = sqrt(1. + ux**2 + uy**2 + uz**2 )
        wn = n_ext > 0.
        jx = zeros(nz+2) ; jy = zeros(nz+2)
        if wn.sum() > 0:
            # jx = zeros(nz+2) ; jy = zeros(nz+2)
            jx[wn] = -(n_ext * ux_ext/gamma)[wn] ;     jy[wn] = -(n_ext * uy_ext/gamma)[wn] ;
            jz = 0.
    else:
        jx = 0. ; jy = 0. ; jz = 0.
    
    dB = dBstep(Ex_ext, Ey_ext)
    dE = dEstep(Bx_ext, By_ext, jx, jy, uz_ext/gamma)
    du, dn, ds, dt_post = dvstep_parabolic(ux_ext, uy_ext, uz_ext, n_ext, s_ext, Ex_ext, Ey_ext, Bx_ext, By_ext)

    
    return dE, dB, du, dn, ds, dt_post
        
def sewerrun():
    
    # E on the edges, B in cell centres (Bz is not evolves, just postulated)
    Bx = -EyA * Bleft(tstart-zlen/2.-dz-z) # minimal z-dz is the coord of the ghost zone
    By = ExA * Bleft(tstart-zlen/2.-dz -z)   
    Ex = ExA * Eleft(tstart-zlen/2.-dz-zhalf) # ghost zone + dz/2.
    Ey = EyA * Eleft(tstart-zlen/2.-dz-zhalf)
    #     Ex = zeros(nz)

    ux = zeros(nz)  ; uy = zeros(nz) ; uz = zeros(nz) ; n = ones(nz) ; s = copy(z) # s is tracer

    if ifnowave:
        uy += exp(-0.5*(z/tpack)**2) * 0.1
        uylist = [] ; uzlist = []
        
    n *= buffermod(z)
    # (z > (-zlen/2.+zbuffer)) & (z < (zlen/2.-zbuffer)) # sin(z * 10.) * 0.1

    # total quantities:
    tlist = []
    mlist = [] # particle mass
    paelist = [] # particle energy
    emelist = [] # EM fields energy
    
    t = tstart ; ctr = 0; plot_ctr = 0 ; hdf_ctr = 0

    uzmax = 0.
    
    while ((t < tmax) & (uzmax < 10.* (EyA/omega0)**2/2.)):
           #           & (abs(Bx).max() < ( 10. * EyA)) & (abs(uy).max() < ( 10. * EyA/omega0)) & (n.max() < 10.) & isfinite(n.max())):

        # first step in RK4
        dE, dB, du, dn1, ds1, dt1 = dsteps(t, (Ex, Ey), (Bx, By), (ux, uy, uz), n, s)
        dEx1, dEy1 = dE    ;   dBx1, dBy1 = dB   ; dux1, duy1, duz1 = du

        # adaptive time step:
        dt = minimum(dtCFL, dt1 * dtfac)
        dt = comm.allreduce(dt, op=MPI.MIN) # calculates one minimal dt
        #if (dt < dtCFL):
        #    print("dt = ", dt)
        
        # second step in RK4
        dE, dB, du, dn2, ds2, dt2 = dsteps(t+dt/2., (Ex+dEx1 * dt/2., Ey+dEy1 * dt/2.),
                                           (Bx+dBx1*dt/2., By+dBy1*dt/2.), (ux+dux1*dt/2., uy+duy1*dt/2., uz+duz1*dt/2.),
                                           n+dn1*dt/2., s + ds1 * dt/2.)
        dEx2, dEy2 = dE    ;   dBx2, dBy2 = dB   ; dux2, duy2, duz2 = du
        
        # third step in RK4
        dE, dB, du, dn3, ds3, dt3 = dsteps(t+dt/2., (Ex+dEx2 * dt/2., Ey+dEy2 * dt/2.),
                                           (Bx+dBx2*dt/2., By+dBy2*dt/2.), (ux+dux2*dt/2., uy+duy2*dt/2., uz+duz2*dt/2.),
                                           n+dn2*dt/2., s + ds2 * dt/2. )
        dEx3, dEy3 = dE    ;   dBx3, dBy3 = dB   ; dux3, duy3, duz3 = du

        # fourth step
        dE, dB, du, dn4, ds4, dt4 = dsteps(t+dt, (Ex+dEx3 * dt, Ey+dEy3 * dt), (Bx+dBx3*dt, By+dBy3*dt),
                                           (ux+dux3*dt, uy+duy3*dt, uz+duz3*dt), n+dn3*dt, s + ds3 * dt)
        dEx4, dEy4 = dE    ;   dBx4, dBy4 = dB   ; dux4, duy4, duz4 = du

        # the actual time step
        Bx = Bx + (dBx1 + 2. * dBx2 + 2. * dBx3 + dBx4) * dt/6. ; By = By + (dBy1 + 2. * dBy2 + 2. * dBy3 + dBy4) * dt/6.
        Ex = Ex + (dEx1 + 2. * dEx2 + 2. * dEx3 + dEx4) * dt/6. ; Ey = Ey + (dEy1 + 2. * dEy2 + 2. * dEy3 + dEy4) * dt/6.
        ux = ux + (dux1 + 2. * dux2 + 2. * dux3 + dux4) * dt/6. ; uy = uy + (duy1 + 2. * duy2 + 2. * duy3 + duy4) * dt/6.
        uz = uz + (duz1 + 2. * duz2 + 2. * duz3 + duz4) * dt/6. ; n = n + (dn1 + 2. * dn2 + 2. * dn3 + dn4) * dt/6.
        s = s + (ds1 + 2. * ds2 + 2. * ds3 + ds4) * dt / 6.

        # filtering negative density
        n = maximum(n, nmin)
        
        # velocity damping:
        
        if ifvdamp:
            '''
            dampcore = exp(-(fsq/fsq.max()*0.1)**2 * dt) + 0.j
            uz_previous = uz
            uz = ifft(fft(uz) * dampcore).real
            uy = ifft(fft(uy) * dampcore).real
            print(abs(uz-uz_previous).max())
            if abs(uz-uz_previous).max() > 1e-5:
                clf()
                plot(z, uz-uz_previous)
                savefig('duz.png')
                tt = input('uz')
            '''    
            # dtcay = 10. * dt
            if ifexpdamp:
                uy *= exp(-maximum(exp(-(z+zlen/2.)/dzcay), 0.)/dtcay)
                uz *= exp(-maximum(exp(-(z+zlen/2.)/dzcay), 0.)/dtcay)
            else:
                uyest = Aleft(t+dt-z-zlen/2.-dz/2.)*EyA
                uzest = uyest**2/2.
                wout = (z< (-zlen/2.+zbuffer*2.)) # |(z>(zlen/2.-zbuffer))
                uy[wout] = uyest[wout] - (uy-uyest)[wout] * exp(- dt / dtcay  * (abs(z[wout])-(zlen/2.-zbuffer*2.))**2)
                uz[wout] = uzest[wout] - (uz-uzest)[wout] * exp(- dt / dtcay * (abs(z[wout])-(zlen/2.-zbuffer*2.))**2)
        
        uzmax = uz.max()
        uzmax = comm.allreduce(uzmax, op=MPI.MAX)
        
        t += dt ; ctr += 1
        hdfflag = (ctr%hdf_alias==0)
        plotflag = ifplot & (ctr%picture_alias==0)
        
        if hdfflag | plotflag:            
            # we need to merge the arrays            
            if crank > first:
                plotpack = {"z": z, "Bx": Bx, "By": By, "Ex": Ex, "Ey": Ey, "ux": ux, "uy": uy, "uz": uz, "n": n, "s": s}
                comm.send(plotpack, dest = first, tag = crank)
            else:
                zplot = z ; Bxplot = Bx ; Eyplot = Ey ; Explot = Ex ; Byplot = By
                uxplot = ux ; uyplot = uy ; uzplot = uz ; nplot = n; splot = s
                #                zplot_half = zplot-dz/2.
                if csize > 1:
                    for k in arange(nc-1)+first+1:
                        plotpack = comm.recv(source = k, tag = k)
                        zplot = concatenate([zplot, plotpack["z"]])
                        Bxplot = concatenate([Bxplot, plotpack["Bx"]])
                        Explot = concatenate([Explot, plotpack["Ex"]])
                        Byplot = concatenate([Byplot, plotpack["By"]])
                        Eyplot = concatenate([Eyplot, plotpack["Ey"]])
                        uxplot = concatenate([uxplot, plotpack["ux"]])
                        uyplot = concatenate([uyplot, plotpack["uy"]])
                        uzplot = concatenate([uzplot, plotpack["uz"]])
                        nplot = concatenate([nplot, plotpack["n"]])
                        splot = concatenate([splot, plotpack["s"]])
                zplot_half = zplot-dz/2.
                # print(size(zplot_half), size(Explot), size(Eyplot))
                # ii = input("E")
                # totals:
                gamma = sqrt(1. + uxplot**2 + uyplot**2 + uzplot**2 )

                mtot = simpson(nplot, x = zplot)
                epatot = 2. * simpson(nplot*(gamma-1.), x = zplot)
                emetot = (simpson(Bxplot**2+Byplot**2, x = zplot) + simpson(Explot**2+Eyplot**2, x = zplot_half))/2.
            
                tlist.append(t)
                mlist.append(mtot)
                paelist.append(epatot)
                emelist.append(emetot)

                if ifnowave:
                    uylist.append(uyplot[(nz*nc)//2])
                    uzlist.append(uzplot[(nz*nc)//2])
            
                print("mlist = ", mtot)
                print("E_PA + E_EM <= ", epatot, " + ", emetot, " = ", epatot + emetot)

                if plotflag: 
                    ww = (abs(uyplot) > 1e-8)
                    clf()
                    fig = figure()
                    if abs(Bxbgd) > 0.:
                        plot(zplot, Bxplot*0. + Bxbgd * Bbuffermod(zplot), 'k:', label = r'$B_x^{\rm bgd}$')
                    plot(zplot, Bxplot+Bxbgd * Bbuffermod(zplot), 'k-', label = r'$B_x$')
                    if abs(Byplot).max() > 0.:
                        plot(zplot, Byplot+Bybgd * Bbuffermod(zplot), 'k:', label = r'$B_y$')
                    plot(zplot_half, Eyplot, 'r-', label = r'$E_y$')
                    if abs(Explot).max() > 0.:
                        plot(zplot_half, Ex, 'r:', label = r'$E_x$')
                    plot(zplot_half[0]-dz, EyA * Eleft(t), 'bo', label = r'$E_y$ BC')
                    xlabel(r'$z$') 
                    title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
                    legend()
                    fig.set_size_inches(12.,5.)
                    savefig('EB{:05d}.png'.format(plot_ctr))
                
                    uytmp = arange(100)/100. * (uyplot.max() - uyplot.min()) + uyplot.min()
                    gammaplot = sqrt(1.+uxplot**2+uyplot**2+uzplot**2)

                    clf()
                    plot(uytmp, uytmp**2/2., 'r-')
                    scatter(uyplot, uzplot, c = zplot)
                    cb = colorbar()
                    cb.set_label(r'$z$')
                    xlabel(r'$u^y$')   ;   ylabel(r'$u^z$')
                    # xlim(-15, -10) ; ylim(-0.1,0.1)
                    title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
                    savefig('GO{:05d}.png'.format(plot_ctr))
                    if ww.sum() > 1:
                        clf()
                        plot(zplot[ww], uyplot[ww], 'k-', label = r'$u^y$')
                        if abs(uxplot).max() > 0.:
                            plot(zplot[ww], uxplot[ww], 'g--', label = r'$u^x$')
                        plot(zplot[ww], uzplot[ww], 'r:', label = r'$u^z$')
                        xlabel(r'$z$') 
                        title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
                        legend()
                        fig.set_size_inches(12.,5.)
                        savefig('uyz{:05d}.png'.format(plot_ctr))
                        clf()
                        plot(zplot[ww], Aleft(t-zplot[ww]-zlen/2.-dz/2.)*EyA, 'b-')
                        # plot(zplot[ww], Aleft(t-splot[ww]-zlen/2.-dz/2.)*EyA, 'b:')
                        plot(zplot[ww], uyplot[ww], 'k.')
                        # plot(zplot[ww], uyplot[ww] * 4., 'k,')
                        xlabel(r'$z$')   ;   ylabel(r'$u^y$') 
                        title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
                        savefig('uGO{:05d}.png'.format(plot_ctr))
                        clf()
                        fig = figure()
                        plot(Aleft(t-zplot[ww]-zlen/2.-dz/2.)*EyA, Aleft(t-zplot[ww]-zlen/2.-dz/2.)*EyA, 'r-')
                        plot(Aleft(t-zplot[ww]-zlen/2.-dz/2.)*EyA, uyplot[ww], 'k.')
                        xlabel(r'$-A$')   ;   ylabel(r'$u^y$') 
                        title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
                        fig.set_size_inches(8.,8.)
                        savefig('AGO{:05d}.png'.format(plot_ctr))
                        clf()
                        plot(zplot[ww], gammaplot[ww]-1., 'bx', label = r'$\gamma-1$')
                        plot(zplot[ww], uzplot[ww] , 'k.', label = r'$u^z$')
                        legend()
                        # yscale('log')
                        xlabel(r'$z$')   ;   ylabel(r'$u^z$') 
                        title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
                        savefig('vGO{:05d}.png'.format(plot_ctr))
                        
                    
                    clf()
                    fig = figure()
                    # plot(zplot, 1.+(Aleft(t-zplot+zplot.min())*EyA)**2/2., 'r:')
                    plot(zplot, nplot, '-k')
                    # cb = colorbar()
                    # cb.set_label(r'$z$')
                    xlabel(r'$z$')   ;   ylabel(r'$n_{\rm p}$') 
                    title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
                    fig.set_size_inches(12.,6.)
                    savefig('n{:05d}.png'.format(plot_ctr))
                    clf()
                    plot(zplot, zplot, 'r:')
                    plot(zplot, splot, '-k')
                    # cb = colorbar()
                    # cb.set_label(r'$z$')
                    xlabel(r'$z$')   ;   ylabel(r'$z0(z, t)$') 
                    title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
                    savefig('sz{:05d}.png'.format(plot_ctr))
                    # currents:
                    jy = nplot * uyplot / gammaplot
                    clf()
                    plot(zplot, jy, 'k.')
                    xlabel(r'$z$')   ;   ylabel(r'$j_y$') 
                    title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
                    savefig('jy{:05d}.png'.format(plot_ctr))
                    close()
                    plot_ctr += 1
            
                # HDF5 dump:
                if hdfflag:
                    if hdf_ctr == 0:
                        hout = hio.fewout_init('fout.hdf5',
                                               {"ifmatter": ifmatter, "ExA": ExA, "EyA": EyA,
                                                "omega0": omega0, "tpack": tpack, "tmid": tmid, "Bz": Bz, "Bx": Bxbgd},
                                               zplot, zhalf = zplot_half)
                        # text output for mass and energies
                        totout = open("totals.dat", 'w+')
                        totout.write("# t -- m -- EM energy -- PA energy \n")

                    hio.fewout_dump(hout, hdf_ctr, t, (Explot, Eyplot), (Bxplot+Bxbgd*Bbuffermod(zplot), Byplot+Bybgd), (uxplot, uyplot, uzplot), nplot)
                    totout.write(str(t)+" "+str(mtot)+" "+str(emetot)+" "+str(epatot)+"\n")
                    totout.flush()
                
                    print("dt = ", dt)                   
                    # comm.bcast()
                    hdf_ctr += 1

    if crank == first:
        hout.close()

        paelist  = asarray(paelist)  ; emelist = asarray(emelist)  ; tlist = asarray(tlist)

        totout.close()
            
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

sewerrun()
