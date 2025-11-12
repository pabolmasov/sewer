import matplotlib
from matplotlib.pyplot import *

# HDF5 io:
import hio

from numpy import *

cmap = 'viridis'

ndigits = 2 # round-off digits TODO: make this automatic

# store all this in the globals:
EyA = 100.0
omega0 = 10.0

tpack = sqrt(6.)
tmid = tpack * 10.
tmax = 3. * tmid

def Aleft(t):
    return -sin(omega0 * t) * exp(-(t/tpack)**2/2.) / omega0

def show_nukeplane(omega0 = 1.0, bgdfield = 0., iflog = True):

    bbgdz = bgdfield
    
    omega, k, datalist = hio.okplane_hread('okplane_Bx.hdf', datanames = ['Bx'])
    bxlist_FF =  datalist[0][:,:]
    # print(type(bxlist_FF[0,0]))
    # print(type(omega[0]))
    # print(type(k[0]))
    
    babs = sqrt(bxlist_FF.real**2 + bxlist_FF.imag**2)
    # fftshift(nu)
    # fftshift(k)
    
    clf()
    fig = figure()
    if iflog:
        babs = log10(ma.masked_array(babs, mask = (babs <= 0.)))
        pcolormesh(omega, k, transpose(babs), vmin = maximum(babs.min(), babs.max()-3.), vmax = babs.max(), cmap = 'hot')
    else:
        pcolormesh(omega, k, transpose(babs), vmin = 0., vmax = babs.max(), cmap = 'hot')
    cb = colorbar()
    cb.set_label(r'$\log_{10} |\tilde{b}_x|$')
    #plot([-f0], [ f0], 'ro', mfc='none')

    
    if abs(bgdfield)>0.01:
        # X and fast modes:
        ktmp = (3. * omega0+2.*bgdfield) * (arange(100)/double(100)-0.5)
        plot(sqrt((1.+bgdfield**2 + ktmp**2 + sqrt((1.+bgdfield**2 + ktmp**2)**2-4.*ktmp**2*bgdfield**2))/2.), ktmp, 'g:', label=r'$\omega_X$')
        plot(sqrt((1.+bgdfield**2 + ktmp**2 - sqrt((1.+bgdfield**2 + ktmp**2)**2-4.*ktmp**2*bgdfield**2))/2.), ktmp, 'g:')
        plot(-sqrt((1.+bgdfield**2 + ktmp**2 + sqrt((1.+bgdfield**2 + ktmp**2)**2-4.*ktmp**2*bgdfield**2))/2.), ktmp, 'g:')
        plot(-sqrt((1.+bgdfield**2 + ktmp**2 - sqrt((1.+bgdfield**2 + ktmp**2)**2-4.*ktmp**2*bgdfield**2))/2.), ktmp, 'g:')
        xlim(-omega0*2.-bgdfield*2., omega0*2.+bgdfield*2.)
        ylim(-omega0*2.-bgdfield*2., omega0*2.+bgdfield*2.)
    else:
        ktmp = 3. * omega0 * (arange(100)/double(100)-0.5)
        plot(sqrt(1.+ktmp**2), ktmp, 'w:')
        plot(-sqrt(1.+ktmp**2), ktmp, 'w:')
        plot(sqrt(1.+ktmp**2), -ktmp, 'w:')
        plot(-sqrt(1.+ktmp**2), -ktmp, 'w:')
    
        xlim(-omega0*3., omega0*3.)
        ylim(-omega0*3., omega0*3.)
    # xlim(omega.min(), omega.max())
    # ylim(k.min(), k.max())
    plot([omega0], [ -omega0], 'bo', mfc='none')
    plot([-omega0], [omega0], 'bo', mfc='none')
    
    #    xlim(1./tmax, 1./dtout)  ;  ylim(1./zlen, 1./dz)
    fig.set_size_inches(15.,10.)
    xlabel(r'$\omega/\omega_{\rm p}$') ; ylabel(r'$k \delta$')    
    savefig('okplane.png')
         

def circorrelate(x, y):

    meanx = x.mean() ; meany = y.mean()
    stdx = x.std() ; stdy = y.std()
    
    nx = size(x)
    r = zeros(nx*2-1)
    
    for k in arange(nx*2-1, dtype = int)-nx:
        y1 = roll(x, k)
        r[k] = ((x-meanx)*(y1-meany)).sum()/stdx/stdy

    return r

def maps_dat(filename = "sewerout.dat", zalias = 1, talias = 1):

    lines = loadtxt(filename)

    t2 = lines[:,0]  ;   z2 = lines[:,1]  ;  q = lines[:,2]

    nt = size(unique(t2))   ; nz = size(unique(z2))

    t2 = reshape(t2, [nt, nz]) ;  z2 = reshape(z2, [nt, nz]) ; q = reshape(q, [nt, nz])

    clf()
    fig = figure()
    pcolormesh(z2[::talias, ::zalias], t2[::talias, ::zalias], q[::talias, ::zalias])
    colorbar()
    plot(z2[0,:], z2[0,:], 'w--')
    xlabel(r'$z$') ; ylabel(r'$\omega_{\rm p} t$')  
    ylim(t2.min(), t2.max())
    fig.set_size_inches(15.,10.)
    savefig('qmap.png')
    

def maps(z, tlist, bxlist, uylist, uzlist, nlist, zalias = 1, talias = 1, zcurrent = None):

    if zcurrent is None:
        zcurrent = z

    s = shape(nlist)
    print("map shape = ", s)
    # if the two species are stored separately
    if s[1] == 2:
        nplist = nlist[0]
        nelist = nlist[1]
    s = shape(uzlist)
    print(s)
    # if the two species are stored separately
    if s[1] == 2:
        uzplist = uzlist[0]
        uzelist = uzlist[1]
    # 2D-visualization
    clf()
    fig = figure()
    pcolormesh(z[::zalias], tlist[::talias], bxlist[::talias, ::zalias])
    colorbar()
    plot(z, z, 'w--')
    ylim(tlist.min(), tlist.max())
    xlabel(r'$z$') ; ylabel(r'$t$')  
    fig.set_size_inches(15.,10.)
    savefig('EBmap.png')
    
    clf()
    fig, ax = subplots(ncols=2, figsize=(10, 6))
    pc1 = ax[0].pcolormesh(z[::zalias], tlist[::talias], uylist[::talias, ::zalias])
    cb1 = fig.colorbar(pc1, ax = ax[0])
    cb1.set_label(r'$u^y$')
    pc2 = ax[1].pcolormesh(z[::zalias], tlist[::talias], uzlist[::talias, ::zalias])
    cb2 = fig.colorbar(pc1, ax = ax[1])
    cb2.set_label(r'$u^z$')
    ax[0].set_xlabel(r'$z_0$') ; ax[1].set_xlabel(r'$z_0$') ; ax[0].set_ylabel(r'$\omega_{\rm p} t$')  
    savefig('uzmap.png')
    
    clf()
    fig, ax = subplots(ncols=2, figsize=(8, 4))
    pc1 = ax[0].pcolormesh(z[::zalias], tlist[::talias], nlist[::talias, ::zalias])
    cb1 = fig.colorbar(pc1, ax = ax[0])
    cb1.set_label(r'$n$')
    pc2 = ax[1].pcolormesh(z[::zalias], tlist[::talias], log10(nlist[::talias, ::zalias]))
    cb2 = fig.colorbar(pc2, ax = ax[1])
    cb2.set_label(r'$\log_{10}n$')
    ax[0].set_xlabel(r'$z$') ; ax[1].set_xlabel(r'$z$') ; ax[0].set_ylabel(r'$\omega_{\rm p} t$')  
    savefig('nmap.png')
    
    close()

def slew(t, z0, z, ay, bx, uy, uz, n, ctr, tmid = tmid):

    zlen = z0.max() - z0.min()
    zcenter = -zlen/2. + t-tmid # used for the simulations without a source
    # zcenter = (zcenter - z0.min()) % zlen + z0.min()
    zcenter_show = minimum(maximum(zcenter, z0.min()), z0.max())
    
    clf()
    fig, ax = subplots(ncols=2, figsize=(8, 4))
    ax[0].plot(z0, z0, 'r:')
    ax[0].plot(z0, z, 'k-')
    ax[0].plot([zcenter_show - 2.*tpack, zcenter_show + 2. * tpack], [zcenter_show - 2.*tpack, zcenter_show - 2. * tpack], 'g:')
    ax[0].plot([zcenter_show - 2.*tpack, zcenter_show + 2. * tpack], [zcenter_show + 2.*tpack, zcenter_show + 2. * tpack], 'g:')
    ax[0].plot([zcenter_show - 2.*tpack, zcenter_show - 2. * tpack], [zcenter_show - 2.*tpack, zcenter_show + 2. * tpack], 'g:')
    ax[0].plot([zcenter_show + 2.*tpack, zcenter_show + 2. * tpack], [zcenter_show- 2.*tpack, zcenter_show + 2. * tpack], 'g:')
    ax[1].plot(z0, z0, 'r:')
    ax[1].plot(z0, z, 'k-')
    ax[1].set_xlim(zcenter_show - 2.*tpack, zcenter_show + 2. * tpack)
    ax[1].set_ylim(zcenter_show - 2.*tpack, zcenter_show + 2. * tpack)    
    ax[0].set_xlabel(r'$z_0$') ; ax[0].set_ylabel(r'$z$')
    ax[1].set_xlabel(r'$z_0$') # ; ax[0].set_ylabel(r'$z$')
    fig.suptitle(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
    savefig('slewzz{:05d}'.format(ctr))

    clf()
    # fig = figure()
    fig, ax = subplots(ncols=2, figsize=(8, 4))
    ax[0].plot(z0, bx, 'k-', label = r'$B_x$')
    ax[0].plot(z0, ay, 'r-', label = r'$E_y$')
    ax[1].plot(z0, bx, 'k-', label = r'$B_x$')
    ax[1].plot(z0, ay, 'r-', label = r'$E_y$')
    ax[0].set_xlabel(r'$z_0$') ; ax[0].set_ylabel(r'$E^y$, $B^x$')
    ax[1].set_xlabel(r'$z_0$') # ; ax[0].set_ylabel(r'$z$')
    ax[1].set_xlim(zcenter_show - 2.*tpack, zcenter_show + 2. * tpack)
    fig.suptitle(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
    ax[0].legend()
    fig.set_size_inches(12.,5.)
    savefig('slewEB{:05d}.png'.format(ctr))

    clf()
    fig, ax = subplots(ncols=2, figsize=(8, 4))
    ax[0].plot(z, uy, 'k-', label = r'$u^y$')
    ax[0].plot(z, uz, 'r:', label = r'$u^z$')
    ax[1].plot(z, uy, 'k-', label = r'$u^y$')
    ax[1].plot(z, uz, 'r:', label = r'$u^z$')
    ax[1].set_xlim(zcenter_show - 2.*tpack, zcenter_show + 2. * tpack)
    ax[0].set_xlabel(r'$z$') ; ax[0].set_ylabel(r'$u^{y, z}$')
    ax[1].set_xlabel(r'$z$') # ; ax[0].set_ylabel(r'$z$')
    #     ax[1].set_xlabel(r'$z$') 
    fig.suptitle(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
    ax[0].legend()
    fig.set_size_inches(12.,5.)
    savefig('slewuyz{:05d}.png'.format(ctr))
    
    clf()
    fig, ax = subplots(ncols=2, figsize=(8, 4))
    ax[0].plot(z0, -Aleft(z0-zcenter)*EyA, 'b-')
    ax[0].plot(z, uy, 'k.')
    ax[1].plot(z0, -Aleft(z0-zcenter)*EyA, 'b-')
    ax[1].plot(z, uy, 'k.')
    ax[1].set_xlim(zcenter_show - 2.*tpack, zcenter_show + 2. * tpack)
    #    ax[1].set_xlabel(r'$z$') 
    ax[0].set_xlabel(r'$z$') ; ax[0].set_ylabel(r'$u^{y}$')
    ax[1].set_xlabel(r'$z$') # ; ax[0].set_ylabel(r'$z$')

    # ax[0].set_ylabel(r'$u^y$')  ;   ax[1].set_ylabel(r'$u^y$')
    fig.suptitle(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
    savefig('slewuGO{:05d}.png'.format(ctr))

    clf()
    fig = figure()
    # plot(zplot, 1.+(Aleft(t-zplot+zplot.min())*EyA)**2/2., 'r:')
    plot(z, n, '-k')
    # cb = colorbar()
    # cb.set_label(r'$z$')
    xlabel(r'$z$')   ;   ylabel(r'$n_{\rm p}$') 
    title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
    fig.set_size_inches(12.,6.)
    savefig('slewn{:05d}.png'.format(ctr))

def slew_eplot(tlist, mlist, emelist, paelist, omega0):
    
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
    
    clf()
    plot(tlist, tlist * 0. + 0.5 / omega0**2)
    plot(tlist, paelist/emelist)
    #    yscale('log')
    xlabel(r'$t$')  ;  ylabel(r'$E$')
    savefig('erat.png')
    
def onthefly(z, zshift, ax0, ay0, az0, bx0, by0, ax, ay, az, bx, by, ux, uy, uz, n, ctr, t, omega = 1.0):

    s = shape(n)
    print(s)
    # if the two species are stored separately
    if s[0] == 2:
        np = n[0]
        ne = n[1]
    
    clf()
    fig, axes = subplots(5)
    # print(shape(z), shape(ax))
    axes[0].plot(z, ax, 'k-')
    axes[0].plot(zshift, ax0, 'r.', markersize = 0.5)
    axes[1].plot(z, ay, 'k-')
    axes[1].plot(zshift, ay0, 'r.', markersize = 0.5)
    axes[2].plot(z, az, 'k-')
    axes[2].plot(zshift, az0, 'r.', markersize = 0.5)
    axes[3].plot(z, bx, 'k-')
    axes[3].plot(zshift, bx0, 'r.', markersize = 0.5)
    axes[4].plot(z, by, 'k-')
    axes[4].plot(zshift, by0, 'r.', markersize = 0.5)
    axes[0].set_ylabel(r'$a_x$')
    axes[1].set_ylabel(r'$a_y$')
    axes[2].set_ylabel(r'$a_z$')
    axes[3].set_ylabel(r'$b_x$')
    axes[4].set_ylabel(r'$b_y$')
    axes[4].set_xlabel(r'$z$')  
    # legend()
    axes[0].set_title(r'$t = '+str(round(t, ndigits))+'$')
    fig.set_size_inches(12.,10.)
    savefig('EB{:05d}.png'.format(ctr))
    clf()
    plot(z, uz, 'k-', label = r'$u^z$')
    # plot(z, ux, 'r:', label = r'$u^x$')
    plot(z, uy, 'g--', label = r'$u^y$')
    xlabel(r'$z$')  ;   ylabel(r'$u^i$') 
    legend()
    title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
    fig.set_size_inches(12.,5.)
    savefig('u{:05d}.png'.format(ctr))

    umax = uy.max()
    utmp = (arange(100)/double(100)-0.5)* umax * 2.
    
    clf()
    plot(utmp, utmp**2/2., 'k-')
    scatter(uy, uz, c = z, cmap = 'hsv')
            # (omega * (z-t)) % (2.*pi), cmap = 'hsv')
    colorbar()
    ylabel(r'$u^z$') ;  xlabel(r'$u^y$')
    title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
    fig.set_size_inches(12.,5.)
    savefig('uu{:05d}.png'.format(ctr))
        
    clf()
    if s[0] == 2:
        plot(z, np, 'k-', label = r'$n_+$')
        plot(z, ne, 'r:', label = r'$n_-$')
    else:
        plot(z, n, 'k-', label = r'$n$')
    xlabel(r'$z$')  ;   ylabel(r'$n$')
    yscale('log')
    legend()
    title(r'$\omega_{\rm p} t = '+str(round(t))+'$')
    fig.set_size_inches(12.,5.)
    savefig('n{:05d}.png'.format(ctr))               
    close()
    
def fourier(k, F_bx, f0, ctr):
    clf()
    plot(k, F_bx.real, 'k.')
    plot(k, F_bx.imag, 'gx')
    plot([f0, f0], [0.,sqrt(F_bx.real**2 + F_bx.imag**2).max()], 'r-')
    plot([-f0, -f0], [0.,sqrt(F_bx.real**2 + F_bx.imag**2).max()], 'r-')
    xlim(-2.*f0, 2.* f0)
    xlabel(r'$f$')  ;   ylabel(r'$\tilde b_x$')
    savefig('f{:05d}.png'.format(ctr))

def energyplot(t, e, prefix = ''):

    clf()
    plot(t, e)
    ylim(e[1:].min(), e[e < (e[1]*100)].max())
    yscale("log")
    xlabel(r'$\omega_{\rm p} t$') 
    ylabel(r'$E$')
    savefig(prefix+'et.png')

    print(prefix+": ", e[-1], " out of ", e[0], "left in the end\n elast / einit = ", e[-1]/e[0]) # systematics
    print(prefix+": ", "energy conservation accuracy: ", e.std()/e[0]) # random error

def Bosc(hname, narray):

    nnarray = size(narray)
    tar = zeros(nnarray) ; uyar = zeros(nnarray) ; uzar = zeros(nnarray)

    for k in arange(nnarray):
        t, z, zhalf, E, B, u, n = hio.fewout_readdump(hname, narray[k])
        ux, uy, uz = u
        nz = size(z)
        tar[k] = t
        uyar[k]  = uy[nz//2]
        uzar[k]  = uz[nz//2]

    Bxbgd = 1.0
    clf()
    fig = figure()
    plot(tar, 0.1 * cos(Bxbgd*tar), 'g:', label = r'$u^y$ expectations')
    plot(tar, -0.1 * sin(Bxbgd*tar), 'b-.', label = r'$u^z$ expectations')
    plot(tar, uyar, 'k.', label = r'$u^y$')
    plot(tar, uzar, 'rx', label = r'$u^z$')
    legend()
    xlabel(r'$t$')  ;  ylabel(r'$u^{y, z}(z=0)$')
    fig.set_size_inches(12., 5.)
    fig.tight_layout()
    savefig('Bosc.png')
    clf()
    plot(0.1 * cos(Bxbgd*tar), -0.1 * sin(Bxbgd*tar), 'k-')
    scatter(uyar, uzar, c = tar*Bxbgd)
    cb = colorbar()
    cb.set_label(r'$\omega_{\rm c} t$')
    xlabel(r'$u^y$')  ;  ylabel(r'$u^{z}$')
    savefig('Bosc_circle.png')
    
def uGOcompare(hname, narr):
    snize = size(narr)

    if snize > 1:
        for k in arange(snize):
            uGOcompare(hname, narr[k])
    else:
        t1, z1, zhalf1, E1, B1, u1, n1 = hio.fewout_readdump(hname, narr)
        Ex1, Ey1 = E1
        Bx1, By1 = B1
        ux1, uy1, uz1 = u1

        ww = (abs(uy1) > 1e-8)
        dz = z1[1]-z1[0]
        zlen = z1[-1] - z1[0]
        
        clf()
        fig = figure()
        plot(z1, uy1, 'ko', label = hname+': $u^y$', mfc = 'none')
        # plot(z1, Aleft(t1-z1+z1.min()-dz/2.)*EyA, 'b-', label = r'$-A$')
        z0 = tpack * 4.-zlen/2.
        # print(z0)
        plot(z1, Aleft((z1-z0-t1+tmid)%zlen)*EyA, 'b-', label = r'$-A$')
        legend()
        title(r'$\omega_{\rm p} t = '+str(round(t1, ndigits))+'$')
        ylim(-EyA/omega0, EyA/omega0)
        xlabel(r'$z$') ; ylabel(r'$u^y$')
        fig.set_size_inches(12.,5.)
        savefig('ucomp{:05d}.png'.format(narr))
        close()
        clf()
        fig = figure()
        plot(z1, uz1, 'ko', label = hname+': $u^z$', mfc = 'none')
        # plot(z1, (Aleft(t1-z1+z1.min()-dz/2.)*EyA)**2/2., 'b-', label = r'$A^2/2$')
        plot(z1, (Aleft((z1-z0-t1+tmid)%zlen)*EyA)**2/2., 'b-', label = r'$A^2/2$')
        legend()
        ylim(-(EyA/omega0)**2/10., (EyA/omega0)**2/2.)
        title(r'$\omega_{\rm p} t = '+str(round(t1, ndigits))+'$')
        xlabel(r'$z$') ; ylabel(r'$u^z$')
        fig.set_size_inches(12.,5.)
        savefig('vcomp{:05d}.png'.format(narr))
        close()
        print(hname, "entry ", narr, " finished")
        
def Hcompare(hname1, hname2, nctr, zoomin = 0., zrange = None):

    if size(nctr) <= 1:
        nctr = (nctr, nctr)
    
    t1, z1, zhalf1, E1, B1, u1, n1 = hio.fewout_readdump(hname1, nctr[0])
    Ex1, Ey1 = E1
    Bx1, By1 = B1
    ux1, uy1, uz1 = u1

    t2, z2, zhalf2, E2, B2, u2, n2 = hio.fewout_readdump(hname2, nctr[1])
    Ex2, Ey2 = E2
    Bx2, By2 = B2
    ux2, uy2, uz2 = u2

    # print("t = ", t1, " = ", t2)
    # print(abs(ux1).max(), abs(ux2).max())
    
    clf()
    fig = figure()
    plot(z1, Bx1, 'k-', label = hname1)
    plot(z2, Bx2, 'r:', label = hname2, linewidth = 2)
    legend()
    if zoomin > 0.:
        zcen = t1 - tmid + z1.min()      
        xlim(zcen - 0.5 * (z1.max()-z1.min())/zoomin, zcen + 0.5 * (z1.max()-z1.min())/zoomin)
        ylim(-sqrt(Bx1**2+Bx2**2).max(), sqrt(Bx1**2+Bx2**2).max())
    title(r'$t = '+str(round(t1, ndigits))+'$')
    xlabel(r'$z$') ; ylabel(r'$B_x$')
    if zrange is not None:
        xlim(zrange[0], zrange[1])
    fig.set_size_inches(12.,5.)
    savefig('compareBx{:05d}_'.format(nctr[0])+'{:05d}'.format(nctr[1]))

    # print((EyA* Aleft(-z1)).max())
    clf()
    fig = figure()
    # plot(z1, sqrt(1.+EyA**2 * Aleft(-z1+t1+z1.min())**2)-1., 'b--', label = r'$\sqrt{1+A^2}-1$')
    plot(z1, uy1, 'k-', label = hname1+'(t = '+str(round(t1, ndigits))+'): $u^y$')
    plot(z2, uy2, 'k:', label = hname2+'(t = '+str(round(t2, ndigits))+': $u^y$', linewidth = 2)
    plot(z1, uz1, 'r-', label = hname1+'(t = '+str(round(t1, ndigits))+': $u^z$')
    plot(z2, uz2, 'r:', label = hname2+'(t = '+str(round(t2, ndigits))+': $u^z$', linewidth = 2)
    if zoomin > 0.:
        xlim(zcen - 0.5 * (z1.max()-z1.min())/zoomin, zcen + 0.5 * (z1.max()-z1.min())/zoomin)
        # xlim(zcen - (zcen-z1.min())/zoomin, zcen + (z1.max()-zcen)/zoomin)
        ylim(-sqrt(uy1**2+uy2**2).max(), sqrt(uy1**2+uy2**2).max())
    legend()
    # title(r'$t = '+str(round(t1, ndigits))+'$')
    xlabel(r'$z$') ; ylabel(r'$u^{y, z}$')
    if zrange is not None:
        xlim(zrange[0], zrange[1])
    fig.set_size_inches(15.,5.)
    savefig('compareu{:05d}_'.format(nctr[0])+'{:05d}'.format(nctr[1]))
    close()
    
def compares(hname1, hname2, narr1, narr2):

    nnarr = size(narr1)
    if size(narr2) != nnarr:
        ii = input("narrs do not match in size")
    
    for k in arange(nnarr):
        Hcompare(hname1, hname2, (narr1[k], narr2[k]), zoomin = 3.)# zrange = [-30., 30.])
    

def compare2d(hname1, hname2, qua = 'Bx'):

    z1, t1, q1 = hio.fewout_readall(hname1, qua = qua,  zalias = 5, talias = 5)
    z2, t2, q2 = hio.fewout_readall(hname2, qua = qua, zalias = 5, talias = 5)

    # print(shape(z1), shape(t1), shape(q1))
    
    clf()
    fig, ax = subplots(ncols=2, figsize=(12, 8))
    ax[0].pcolormesh(z1, t1, transpose(q1), shading = 'nearest')
    ax[0].plot(z1, z1+tmid, 'w:')
    ax[1].pcolormesh(z2, t2, transpose(q2), shading = 'nearest')
    ax[1].plot(z2, z2+tmid, 'w:')
    ax[0].set_xlabel(r'$z$') ;     ax[0].set_ylabel(r'$t$')
    ax[1].set_xlabel(r'$z$') ;     ax[1].set_ylabel(r'$t$')
    ax[0].set_ylim(maximum(t1.min(), t2.min()), minimum(t1.max(), t2.max()))
    ax[1].set_ylim(maximum(t1.min(), t2.min()), minimum(t1.max(), t2.max()))
    savefig('compare2d_'+qua+'.png')

def vamps(hname, narr):

    snarr = size(narr)

    if snarr <= 1:
        # reading a single file
        t1, z1, zhalf1, E1, B1, u1, n1 = hio.fewout_readdump(hname, narr)
        Ex1, Ey1 = E1
        Bx1, By1 = B1
        ux1, uy1, uz1 = u1

        uymin = uy1.min() ; uymax = uy1.max()
        uzmin = uz1.min() ; uzmax = uz1.max()
        
        return t1, (uymin, uymax), (uzmin, uzmax)
    else:

        t = zeros(snarr) ; uymin = zeros(snarr) ; uymax = zeros(snarr)
        uzmin = zeros(snarr) ; uzmax = zeros(snarr)
        
        for k in arange(snarr):
            ttmp, vy, vz = vamps(hname, narr[k])
            t[k] = ttmp
            uymin[k] = vy[0]   ; uymax[k] = vy[1]
            uzmin[k] = vz[0]   ; uzmax[k] = vz[1]
            print("uy = ", uymin[k], '..', uymax[k])

        clf()
        fig, ax = subplots(ncols=2, figsize=(12, 8))
        ax[0].errorbar(t, (uymin+uymax)/2., yerr = (uymax-uymin)/2., fmt = '')
        ax[1].errorbar(t, (uzmin+uzmax)/2., yerr = (uzmax-uzmin)/2., fmt = '')
        ax[0].set_xlabel(r'$t$') ;     ax[0].set_ylabel(r'$u^y$')
        ax[1].set_xlabel(r'$t$') ;     ax[1].set_ylabel(r'$u^y$')
        savefig('vamps.png')
        
