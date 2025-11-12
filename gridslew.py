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

# time and efficiency measurement
from timer import Timer

# simulating a wave moving to the right along z in pair relativistic plasma
# E, B, and v are allowed to have all the three components

# physical switches:
ifmatter = True
ifonedirection = False
ifzclean = True
iflnn = False
ifgridn = False
ifsource = False
ifvdamp = False
ifassumemonotonic = False

# mesh:
nz = 8192
zlen = 60.
z0 = (arange(nz) / double(nz) - 0.5) * zlen # centres of Euler cells
dz = z0[1] - z0[0]
print("dz = ", dz)
z0half = z0 - dz/2. # (z[1:]+z[:-1])/2. # faces of Euler cells
z0_ext = concatenate([[z0[0]-dz], z0, [z0[-1]+dz]]) # Euler cells including the ghost zones
z0half_ext = concatenate([[z0half[0]-dz], z0half, [z0half[-1]+dz]]) # Euler cells including the ghost zones

# time
dtCFL = dz * 0.1 # CFL in 1D should be not very small
dtfac = 0.01
# dtout = 0.01
ifplot = True
hdf_alias = 1000
picture_alias = 10
dtout = 0.1
tstart = 0.

# injection:
ExA = 0.0
EyA = 100.0
omega0 = 10.0
tpack = sqrt(6.)
tmid = tpack * 10. # the passage of the wave through z=0
tmax = zlen + tmid-tpack

# density floor
nlim = 1e-3

# maximal number of monotonic chunks
nmonmax = 100

# decay parameters
dtcay = dtCFL * 10.0
dzcay = 10.0
zbuffer = 5.0
sclip = 2.0

# background magnetic field
Bxbgd = 0.0
Bybgd = 0.0
Bzbgd = 0.0

def Aleft(t):
    return -sin(omega0 * (t -tmid - dz/2.)) * exp(-((t - dz/2.-tmid)/tpack)**2/2.) / omega0

def Eleft(t):
    return (cos(omega0 * (t-tmid)) - (t-tmid)/(omega0*tpack**2) * sin(omega0 * (t-tmid))) * exp(-((t-tmid)/tpack)**2/2.)

def Bleft(t):
    return Eleft(t - dz/2.)
# sin(omega0 * (t-tmid+dz/2.)) * exp(-((t+dz/2.-tmid)/tpack)**2/2.)


def zclean(z, uz):
    # 1 calculates intra-point dispersion
    # dd = abs(z[1:]-z[:-1]).mean()
    
    # 2 outliers from the average value
    w = abs(z[1:-1]-((z[2:]+z[:-2])/2.)) > (sclip * dz)

    if w.sum() > 0:
        # print("dd = ", dd)
        print(w.sum(), "point(s) replaced: ", (z[1:-1])[w])
        (z[1:-1])[w] = ((z[2:]+z[:-2])/2.)[w]
        (uz[1:-1])[w] = ((uz[2:]+uz[:-2])/2.)[w]
        print(w.sum(), "point(s) replaced (new value(s)): ", (z[1:-1])[w])
        # ii = input("d")
    return z, uz

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


def jacoden(z, n0):
    # density as a Jacobian (assuming n0 = 1.0)
    return dz * n0 / maximum(abs(roll(z, 1)- z), abs(z -roll(z, -1)))
# sqrt((roll(z, 1)- z)**2 +  (z -roll(z, -1))**2 + (0.1*dz)**2)

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
    
    
def dsteps(t, z, E, B, u, n0 = None, thetimer = None):
    # one RK4 step
    # z is the grid of the Lagrangian cell coordinates

    nmon = 1 # number of monotonic regions (works only with the feedback)
    
    # essential grid quantities:
    Ex, Ey = E
    Bx, By = B
    ux, uy, uz = u

    if n0 is None:
        n0 = ones(nz)

    # ghost zones:
    if thetimer is not None:
        thetimer.start_comp("BC")
    Ex0 = ExA * Eleft(t)
    Ey0 = EyA * Eleft(t)
    Bx0 = EyA * Bleft(t)
    By0 = -ExA * Bleft(t)
    Ex1 = By[-1]
    Ey1 = - Bx[-1]
    Bx1 = Bx[-1]
    By1 = By[-1]
    uz0 = 0. ; uz1 = uz[-1]
    uy0 = 0. ; uy1 = uy[-1]
    uz0 = (ExA**2 + EyA**2) * Aleft(t)**2/2.
    uy0 = EyA * Aleft(t) # - Bxbgd * dz

    # extended arrays:
    Ex_ext = concatenate([[Ex0]] + [Ex] + [[Ex1]]) # - Bybgd + Ex[-1])/2.]] )
    Ey_ext = concatenate([[Ey0]] + [Ey] +  [[Ey1]]) # + Bxbgd +Ey[-1])/2.]] )
    Bx_ext = concatenate([[Bx0]] + [Bx] + [[Bx1]] )
    By_ext = concatenate([[By0]] + [By] +  [[By1]])
    z_ext = concatenate([[z[0]-dz]] + [z] +  [[z[-1]+dz]])
    if thetimer is not None:
        thetimer.stop_comp("BC")
    
    n =  jacoden(z_ext, n0)[1:-1] # lab-frame density n\gamma

    gamma = sqrt(1.+ux**2+uy**2+uz**2)
    vx = ux/gamma ; vy = uy/gamma ; vz = uz/gamma

    jx = z0half_ext * 0. ; jy =  z0half_ext * 0. ; vz_ext =  z0half_ext * 0.
    
    # two mappings
    if ifmatter:
        # mapping currents from Lagrangian to Eulerian
        if thetimer is not None:
            thetimer.start_comp("currents")
        wg = z[1:] < z[:-1] # make this condition harder? 
        if (wg.sum() <= 2) | ifassumemonotonic:
            # jxfun = interp1d(z, -n * vx, bounds_error = False, fill_value = 0.) !!! just because we do not really have vx
            jyfun = interp1d(z, -n * vy, bounds_error = False, fill_value = 0., kind='cubic')
            # jx = jxfun(z0half_ext) ;
            jy = jyfun(z0half_ext)
            jx = jy * 0.
            vzfun = interp1d(z, vz, bounds_error = False, fill_value = (uz0, uz1), kind='cubic')
            vz_ext = vzfun(z0half_ext)
        else:
            # there are non-monotonic regions
            wmonotonic = monotonic_split(z)
            nchunks = len(wmonotonic)
            nmon = nchunks
            jx = z0half_ext * 0. ;  jy = z0half_ext * 0. ; vz_ext = z0half_ext * 0. ; vz_norm = z0half_ext * 0.
            # print(nchunks, " monotonic chunks, sizes: ")
            for k in arange(nchunks):
                w = (z0half_ext > z[wmonotonic[k]].min()) * (z0half_ext < z[wmonotonic[k]].max())
                if w.sum() > 2:
                    # jxfun = interp1d(z[wmonotonic[k]], -(n * vx)[wmonotonic[k]], bounds_error = False, fill_value = 0.)  !!! just because we do not really have vx
                    jyfun = interp1d(z[wmonotonic[k]], -(n * vy)[wmonotonic[k]], bounds_error = False, fill_value = 0.)
                    vzfun = interp1d(z[wmonotonic[k]], (vz*n)[wmonotonic[k]], bounds_error = False, fill_value = (uz0, uz1))
                    nfun = interp1d(z[wmonotonic[k]], n[wmonotonic[k]], bounds_error = False, fill_value = (1., 1.))
                    # jx[w] += jxfun(z0half_ext[w])
                    jy[w] += jyfun(z0half_ext[w])                    
                    vz_ext[w] += vzfun(z0half_ext[w]) ; vz_norm[w] += nfun(z0half_ext[w])
                    # print("size ", w.sum())
            vz_ext[vz_norm > 0.] /= vz_norm[vz_norm > 0.]
        if thetimer is not None:
            thetimer.stop_comp("currents")
                    
    # mapping fields from Eulerian grid to Lagransian
    if thetimer is not None:
        thetimer.start_comp("Maxwell")
    # Exfun = interp1d(z0half, Ex, bounds_error = False, fill_value = (Ex0, Ex1)) !!! no fields or motions in x direction
    Eyfun = interp1d(z0half, Ey, bounds_error = False, fill_value = (Ey0, Ey1), kind='cubic')
    Bxfun = interp1d(z0, Bx + Bxbgd, bounds_error = False, fill_value = (Bx0 + Bxbgd, Bx1 + Bxbgd), kind='cubic')
    # Byfun = interp1d(z0, By + Bybgd, bounds_error = False, fill_value = (By0 + Bybgd, By1 + Bybgd)) !!! assuming no B field in y direction
    
    # Maxwell equations:
    dB = dBstep(Ex_ext, Ey_ext)
    dE = dEstep(Bx_ext, By_ext, jx, jy, vz_ext)
    if thetimer is not None:
        thetimer.stop_comp("Maxwell")

    dux =  vy * Bzbgd # - vz * Byfun(z) # !!! Ex and BY excluded
    duy = Eyfun(z) + vz * Bxfun(z) - vx * Bzbgd
    duz = -vy * Bxfun(z)  # vx * Byfun(z) - vy * Bxfun(z)     !!! By excluded
    
    dzz = vz 
    
    # dt_n = 0.5/maximum(abs(dn).max(), 1./tmax)    
    dt_post = 1./maximum(abs(duy), abs(duz)).max()
    
    return dE, dB, (dux, duy, duz), dzz, dt_post, nmon

def sewerrun():

    thetimer = Timer(["total", "step", "io"],
                     ["BC", "currents", "Maxwell", "RKstep", "cleaning"])
    if thetimer is not None:
        thetimer.start("total")
        # thetimer.start("io")
    
    f = fftfreq(nz, d = dz / (2. * pi)) # Fourier mesh
    z = copy(z0)

    # initial conditions:
    Bx = -EyA * Bleft(tstart-zlen/2.-dz-z0) # minimal z-dz is the coord of the ghost zone
    By = ExA * Bleft(tstart-zlen/2.-dz -z0)   
    Ex = ExA * Eleft(tstart-zlen/2.-dz-z0half) # ghost zone + dz/2.
    Ey = EyA * Eleft(tstart-zlen/2.-dz-z0half)

    # 4-velocity
    ux = 0.*z  ;    uy = 0. * z
    uz = uy**2/2.
    n0 = ones(nz+2) * 1.0 * utile.bufferfun((z0_ext - (z0_ext.min()+zbuffer))/zbuffer) # density ; let us keep it unity, meaning time is in omega_p units. Lengths are internally in c/f = 2pi c / omega units, that allows a simpler expression for d/dz 

    dt = dtCFL 
    t = 0. 
    ctr = 0
    tstore = 0.
    tlist = []
    bxlist = []
    uzlist = []
    uylist = []
    nlist = []
    mlist = [] # particle mass
    paelist = [] # particle energy
    emelist = [] # EM fields energy    

    fout_energy = open('slew_energy.dat', 'w+')
    fout = open('slewout.dat', 'w+')
    fout.write('# t -- z -- Bx \n')
    fout_energy.write('# t -- mass -- EM energy -- PA energy\n')

    if ifmatter:
        fout_chunks = open('monout.dat', 'w+')
        fout_chunks.write('# t -- Nchunks1 -- Nchunks2 -- Nchunks3 -- Nchunks4 \n')
    
    thetimer.start("io")
    plots.slew(0., z0, z, Ey, Bx, uy, uz, n0[1:-1], -1)
    thetimer.stop("io")
    #     ii = input('p')
    nmon = 0
    
    while (t < tmax) & (nmon < nmonmax):        
        thetimer.start("step")
        if t > (tstore + dtout - dt):
            # save previous values
            Ex_prev = Ex ;    Ey_prev = Ey  
            Bx_prev = Bx ;    By_prev = By  
            ux_prev = ux ;    uy_prev = uy  ;    uz_prev = uz
            z_ext = concatenate([[z[0]-dz]] + [z] +  [[z[-1]+dz]])
            n_prev = jacoden(z_ext, n0)[1:-1]
            gamma_prev = sqrt(1.+ux_prev**2+uy_prev**2+uz_prev**2)
            
        # TODO: make it dictionaries or structures    
        
        dE, dB, du, dzz1, dt1, nmon1 = dsteps(t, z, (Ex, Ey), (Bx, By), (ux, uy, uz), n0 = n0, thetimer = thetimer)
        dEx1, dEy1 = dE    ;   dBx1, dBy1 = dB   ; dux1, duy1, duz1 = du
        dt = minimum(dtCFL, dt1 * dtfac)# adaptive time step        
        
        # second step in RK4
        dE, dB, du, dzz2, dt2, nmon2 = dsteps(t+dt/2., z + dzz1 * dt/2., (Ex+dEx1 * dt/2., Ey+dEy1 * dt/2.),
                                           (Bx+dBx1*dt/2., By+dBy1*dt/2.), (ux+dux1*dt/2., uy+duy1*dt/2., uz+duz1*dt/2.),
                                           n0 = n0, thetimer = thetimer)
        dEx2, dEy2 = dE    ;   dBx2, dBy2 = dB   ; dux2, duy2, duz2 = du
        
        # third step in RK4
        dE, dB, du, dzz3, dt3, nmon3 = dsteps(t+dt/2., z + dzz2 * dt/2., (Ex+dEx2 * dt/2., Ey+dEy2 * dt/2.),
                                           (Bx+dBx2*dt/2., By+dBy2*dt/2.), (ux+dux2*dt/2., uy+duy2*dt/2., uz+duz2*dt/2.),
                                           n0 = n0, thetimer = thetimer)
        dEx3, dEy3 = dE    ;   dBx3, dBy3 = dB   ; dux3, duy3, duz3 = du

        # fourth step
        dE, dB, du, dzz4, dt4, nmon4 = dsteps(t+dt, z + dzz3 * dt, (Ex+dEx3 * dt, Ey+dEy3 * dt),
                                           (Bx+dBx3*dt, By+dBy3*dt), (ux+dux3*dt, uy+duy3*dt, uz+duz3*dt),
                                           n0 = n0, thetimer = thetimer)
        dEx4, dEy4 = dE    ;   dBx4, dBy4 = dB   ; dux4, duy4, duz4 = du
        

        nmon = maximum(nmon1, maximum(nmon2, maximum(nmon3, nmon4)))
        
        # time step:
        thetimer.start_comp("RKstep")
        Bx = Bx + (dBx1 + 2. * dBx2 + 2. * dBx3 + dBx4) * dt/6. ; By = By + (dBy1 + 2. * dBy2 + 2. * dBy3 + dBy4) * dt/6.
        Ex = Ex + (dEx1 + 2. * dEx2 + 2. * dEx3 + dEx4) * dt/6. ; Ey = Ey + (dEy1 + 2. * dEy2 + 2. * dEy3 + dEy4) * dt/6.

        ux += (dux1 + dux2 * 2. + dux3 * 2. + dux4) / 6. * dt
        uy += (duy1 + duy2 * 2. + duy3 * 2. + duy4) / 6. * dt
        uz += (duz1 + duz2 * 2. + duz3 * 2. + duz4) / 6. * dt
        z  += (dzz1 + dzz2 * 2. + dzz3 * 2. + dzz4) / 6. * dt
        # F_n += (dF_n1 + dF_n2 * 2. + dF_n3 * 2. + dF_n4) / 6. * dt
        t += dt
        thetimer.stop_comp("RKstep")

        # z cleaning
        thetimer.start_comp("cleaning")
        if ifzclean:
            znew, uznew = zclean(z, uz)
            z = znew; uz = uznew
        
        # velocity damping
        if ifvdamp:
            dampfactor = exp(-dt/dtcay  * exp(-(t-tpack)/dzcay - (z0-z0.min()-tpack - zbuffer)/dzcay)) 
            # z = z + (z0-z) * (1.-dampfactor)
            uz *= dampfactor
            uy *= dampfactor
        thetimer.stop_comp("cleaning")
        thetimer.stop("step")
               
        if t > (tstore + dtout):
            thetimer.start("io")
            print("t = ", t)
            if nmon > 1:
                print(nmon, " monotonic regions\n")

            z_ext = concatenate([[z[0]-dz]] + [z] +  [[z[-1]+dz]])
            n = jacoden(z_ext, n0)[1:-1]
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

            hio.fewout_dump(hout, ctr, t, (Ex, Ey), (Bx+Bxbgd, By+Bybgd), (ux, uy, uz), n, zcurrent  = z)
            
            # ASCII output
            for k in arange(size(Bx)):
                fout.write(str(t) + ' ' + str(z[k]) + ' ' + str(Bx[k])+'\n')
            fout.flush()

            mtot = simpson(n0[1:-1], x = z0)
            epatot = simpson((n0[1:-1] * (gamma-1.)).real, x = z0)
            emetot = (simpson(Bx**2+By**2, x = z0) + simpson(Ex**2+Ey**2, x = z0half))/2.

            fout_energy.write(str(t) + ' ' + str(mtot) + ' ' + str(emetot) + ' ' + str(epatot) + '\n')
            fout_energy.flush()
            if ifmatter:
                fout_chunks.write(str(t)+' '+str(nmon1)+' '+str(nmon2)+' '+str(nmon3)+' '+str(nmon4)+'\n')
                fout_chunks.flush()
                
            # tlist.append(t)
            mlist.append(mtot)
            paelist.append(epatot)
            emelist.append(emetot)            
            
            if ctr%picture_alias==0:
                # plots.onthefly(z, (z+zlen/2.+t)%zlen-zlen/2., ax0, ay0, az0, bx0 + Bxbgd, by0 + Bybgd, ax, ay, az, bx + Bxbgd, by + Bybgd, ux, uy, uz, n/gamma, ctr, t)
                plots.slew(t, z0, z, Ey, Bx+Bxbgd, uy, uz, n, ctr, tmid = tmid)
                
            tlist.append(tstore)
            bxlist.append(Bx_prev.real + ((tstore-(t-dt))/dt) * (Bx-Bx_prev).real + Bxbgd)
            # print(len(Fbxlist))
            uylist.append(uy_prev.real +  ((tstore-(t-dt))/dt) * (uy-uy_prev))
            uzlist.append(uz_prev.real +  ((tstore-(t-dt))/dt) * (uz-uz_prev))
            nlist.append((n_prev/gamma_prev).real +  ((tstore-(t-dt))/dt) * (n/gamma - n_prev/gamma_prev))
            tstore += dtout
            
            thetimer.stop("io")
            if ctr%picture_alias==0:
                thetimer.stats("step")
                thetimer.stats("io")
                thetimer.comp_stats()
                thetimer.purge_comps()           

            ctr += 1
    fout.close()   ; fout_energy.close()
    hout.close()
    if ifmatter:
        fout_chunks.close()
    
    tlist = asarray(tlist)
    bxlist = asarray(bxlist)
    uylist = asarray(uylist)
    uzlist = asarray(uzlist)
    nlist = asarray(nlist)
    mlist = asarray(mlist)
    emelist = asarray(emelist)
    paelist = asarray(paelist)

    print(shape(uylist))
    
    nt = size(tlist)

    #    print(tlist[1]-tlist[0], dtout)
    print("dtout = ", (tlist[1:]-tlist[:-1]).min(), (tlist[1:]-tlist[:-1]).max())

    # final mass and energy plots
    if ifplot:
        plots.maps(z0, tlist, bxlist, uylist, uzlist, nlist, zalias = 4, talias = 1, zcurrent = z)

        # make a nukeplane!
        f = fftfreq(nz, d = dz / (2. * pi)) # Fourier mesh in z
        ofreq = fftfreq(size(tlist), dtout)
        bxlist_FF = fft(fft(bxlist, axis = 0), axis = 1) # inner is omega, outer is time

        bxlist_FF = fftshift(bxlist_FF)
        ofreq = fftshift(ofreq)
        f = fftshift(f)
        
        hio.okplane_hout(ofreq*2. * pi, f, bxlist_FF, hname = 'okplane_Bx.hdf', dataname = 'Bx')
        plots.show_nukeplane(omega0 = omega0, bgdfield = Bxbgd)
        
        plots.slew_eplot(tlist, mlist, emelist, paelist, omega0)
                
# plots.show_nukeplane(omega0 = omega0, bgdfield = Bxbgd)
sewerrun()
# ffmpeg -f image2 -r 20 -pattern_type glob -i 'EB*.png' -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  -pix_fmt yuv420p -b 8192k EB.mp4
# uGOcompare('sout_A2_nofeed.hdf5', arange(1000))
