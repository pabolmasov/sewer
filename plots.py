import matplotlib
from matplotlib.pyplot import *

# HDF5 io:
import hio

# utilities:
import utile

from numpy import *
from scipy.integrate import simpson

cmap = 'viridis'

ndigits = 2 # round-off digits TODO: make this automatic

# store all this in the globals:
EyA = 100.0
omega0 = 10.0

tpack =  sqrt(6.) * 2.
tmid = tpack * 10.
tmax = 3. * tmid

def Aleft(t, omega0 = omega0, tpack = tpack):
    return -sin(omega0 * t) * exp(-(t/tpack)**2/2.)

def show_nukeplane(omega0 = 1.0, bgdfield = 0., iflog = True, ddir = './'):

    bbgdz = bgdfield
    
    omega, k, datalist = hio.okplane_hread(ddir + '/okplane_Bx.hdf', datanames = ['Bx'])
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
    savefig(ddir + '/okplane.png')
         

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
    

def maps(z, tlist, bxlist, uylist, uzlist, nlist, zalias = 1, talias = 1, zcurrent = None, ddir = './'):

    #if zcurrent is None:
    #    zcurrent = z

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
    pcolormesh(z[::zalias], tlist[::talias], bxlist[::talias, ::zalias], vmin = -EyA, vmax = EyA)
    colorbar()
    plot(z, z, 'w--')
    ylim(tlist.min(), tlist.max())
    xlabel(r'$z$') ; ylabel(r'$t$')  
    fig.set_size_inches(15.,10.)
    savefig(ddir + '/EBmap.png')
    
    clf()
    fig, ax = subplots(ncols=2, figsize=(10, 6))
    pc1 = ax[0].pcolormesh(z[::zalias], tlist[::talias], uylist[::talias, ::zalias])
    cb1 = fig.colorbar(pc1, ax = ax[0])
    cb1.set_label(r'$u^y$')
    pc2 = ax[1].pcolormesh(z[::zalias], tlist[::talias], uzlist[::talias, ::zalias])
    cb2 = fig.colorbar(pc1, ax = ax[1])
    cb2.set_label(r'$u^z$')
    ax[0].set_xlabel(r'$z_0$') ; ax[1].set_xlabel(r'$z_0$') ; ax[0].set_ylabel(r'$\omega_{\rm p} t$')  
    savefig(ddir + '/uzmap.png')

    if zcurrent is not None:

        tlist2 = copy(uylist) ; z2 = copy(uylist)
        for k in arange(size(tlist)):
            tlist2[k, :] = tlist[k]
            z2[k, :] = z[:]
        
        clf()
        fig, ax = subplots(ncols=2, figsize=(10, 6))
        pc1 = ax[0].pcolormesh(zcurrent[::talias, ::zalias], tlist2[::talias, ::zalias], uylist[::talias, ::zalias])
        cb1 = fig.colorbar(pc1, ax = ax[0])
        cb1.set_label(r'$u^y$')
        ax[0].contour(zcurrent[::talias, ::zalias], tlist2[::talias, ::zalias], z2[::talias, ::zalias], colors= 'w')
        pc2 = ax[1].pcolormesh(zcurrent[::talias, ::zalias], tlist2[::talias, ::zalias], uzlist[::talias, ::zalias])
        cb2 = fig.colorbar(pc1, ax = ax[1])
        cb2.set_label(r'$u^z$')
        ax[1].contour(zcurrent[::talias, ::zalias], tlist2[::talias, ::zalias], z2[::talias, ::zalias], colors= 'w')
        ax[0].set_xlabel(r'$z$') ; ax[1].set_xlabel(r'$z$') ; ax[0].set_ylabel(r'$\omega_{\rm p} t$')  
        savefig(ddir + '/Luzmap.png')
        
        clf()
        fig, ax = subplots(ncols=2, figsize=(15, 8))
        pc1 = ax[0].pcolormesh(zcurrent[::talias, ::zalias], tlist2[::talias, ::zalias], log10(nlist[::talias, ::zalias]), vmin = log10(maximum(nlist.min(), 0.1)))
        cb1 = fig.colorbar(pc1, ax = ax[0])
        cb1.set_label(r'$n$')
        pc2 = ax[1].pcolormesh(z[::zalias], tlist2[::talias, ::zalias], log10(nlist[::talias, ::zalias]), vmin = log10(maximum(nlist.min(), 0.1)))
        cb2 = fig.colorbar(pc2, ax = ax[1])
        cb2.set_label(r'$\log_{10}n$')
        ax[0].set_xlabel(r'$z$') ; ax[1].set_xlabel(r'$z_0$') ; ax[0].set_ylabel(r'$\omega_{\rm p} t$')  
        savefig(ddir + '/nmap.png')
    
    close()

def slew(t, z0, z, ay, bx, uy, uz, n, ctr, ddir = './', EyA = EyA, omega0 = omega0, tpack = tpack, tmid = tmid):

    zlen = z0.max() - z0.min()
    zcenter = -zlen/2. + t-tmid # used for the simulations without a source
    # zcenter = (zcenter - z0.min()) % zlen + z0.min()
    zcenter_show = minimum(maximum(zcenter, z0.min()), z0.max())

    wzcur = (z0 > (zcenter_show - 2.*tpack))*( z< zcenter_show + 2. * tpack)

    if wzcur.sum() > 0:
        zcur1 = z[(z0 > (zcenter_show - 2.*tpack))*( z< zcenter_show + 2. * tpack)].min()
        zcur2 = z[(z0 > (zcenter_show - 2.*tpack))*( z< zcenter_show + 2. * tpack)].max()
    else:
        zcur1 = zcenter_show - 2.*tpack
        zcur2 = zcenter_show + 2.*tpack
    clf()
    fig, ax = subplots(ncols=2, figsize=(8, 4))
    ax[0].plot(z0, z0, 'r:')
    ax[0].plot(z0, z, 'k-')
    ax[0].plot([zcenter_show - 2.*tpack, zcenter_show + 2. * tpack], [zcur1, zcur1], 'g:')
    ax[0].plot([zcenter_show - 2.*tpack, zcenter_show + 2. * tpack], [zcur2, zcur2], 'g:')
    ax[0].plot([zcenter_show - 2.*tpack, zcenter_show - 2. * tpack], [zcur1, zcur2], 'g:')
    ax[0].plot([zcenter_show + 2.*tpack, zcenter_show + 2. * tpack], [zcur1, zcur2], 'g:')
    ax[1].plot(z0, z0, 'r:')
    ax[1].plot(z0, z, 'k-')
    ax[1].set_xlim(zcenter_show - 2.*tpack, zcenter_show + 2. * tpack)
    ax[1].set_ylim(zcur1, zcur2)    
    ax[0].set_xlabel(r'$z_0$') ; ax[0].set_ylabel(r'$z$')
    ax[1].set_xlabel(r'$z_0$') # ; ax[0].set_ylabel(r'$z$')
    fig.suptitle(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
    savefig(ddir + '/slewzz{:05d}'.format(ctr))

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
    savefig(ddir + '/slewEB{:05d}.png'.format(ctr))

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
    savefig(ddir + '/slewuyz{:05d}.png'.format(ctr))
    
    clf()
    fig, ax = subplots(ncols=3, figsize=(12, 4))
    ax[0].plot(z0, -Aleft(z0-zcenter, omega0 = omega0, tpack = tpack)*EyA/omega0, 'b-')
    ax[0].plot(z, uy, 'k.')
    ax[1].plot(z0, -Aleft(z0-zcenter, omega0 = omega0, tpack = tpack)*EyA/omega0, 'b-')
    ax[1].plot(z, uy, 'k.')
    ax[2].plot(z0, -Aleft(z0-zcenter, omega0 = omega0, tpack = tpack)*EyA/omega0, 'b-')
    ax[2].plot(z, uy, 'k.')
    ax[1].set_xlim(zcenter_show - 2.*tpack, zcenter_show + 2. * tpack)
    ax[2].set_xlim(zcenter_show - 2.*2.*pi/omega0, zcenter_show + 2. * 2.*pi/omega0)
    ax[0].set_xlabel(r'$z$') ; ax[0].set_ylabel(r'$u^{y}$')
    ax[1].set_xlabel(r'$z$') # ; ax[0].set_ylabel(r'$z$')
    ax[2].set_xlabel(r'$z$') # ; ax[0].set_ylabel(r'$z$')
    fig.suptitle(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
    savefig(ddir + '/slewuGO{:05d}.png'.format(ctr))
    clf()
    fig, ax = subplots(ncols=3, figsize=(12, 4))
    ax[0].plot(z0, (Aleft(z0-zcenter, omega0 = omega0, tpack = tpack)*EyA/omega0)**2/2., 'b-')
    ax[0].plot(z, uz, 'k.')
    ax[1].plot(z0, (Aleft(z0-zcenter, omega0 = omega0, tpack = tpack)*EyA/omega0)**2/2., 'b-')
    ax[1].plot(z, uz, 'k.')
    ax[2].plot(z0, (Aleft(z0-zcenter, omega0 = omega0, tpack = tpack)*EyA/omega0)**2/2., 'b-')
    ax[2].plot(z, uz, 'k.')
    ax[1].set_xlim(zcenter_show - 2.*tpack, zcenter_show + 2. * tpack)
    ax[2].set_xlim(zcenter_show - 2.*2.*pi/omega0, zcenter_show + 2. * 2.*pi/omega0)
    ax[0].set_xlabel(r'$z$') ; ax[0].set_ylabel(r'$u^{z}$')
    ax[1].set_xlabel(r'$z$') # ; ax[0].set_ylabel(r'$z$')
    ax[2].set_xlabel(r'$z$') # ; ax[0].set_ylabel(r'$z$')
    fig.suptitle(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
    savefig(ddir + '/slewvGO{:05d}.png'.format(ctr))

    clf()
    # fig = figure()
    fig, ax = subplots(ncols=2, figsize=(10, 4), width_ratios = (2,1))
    # plot(zplot, 1.+(Aleft(t-zplot+zplot.min())*EyA)**2/2., 'r:')
    ax[0].plot(z, 1.+uz, 'b-', label = r'$1+u^z$')
    ax[0].plot(z, n, '.k', label = r'$\gamma n$', mfc = 'none')
    ax[0].legend()
    ax[1].plot(z, 1.+uz, 'b-', label = r'$1+u^z$')
    ax[1].plot(z, n, '.k', label = r'$\gamma n$', mfc = 'none')
    ax[0].set_xlabel(r'$z$') ; ax[1].set_xlabel(r'$z$') ; ax[0].set_ylabel(r'$n$, $1+u^z$')
    ax[1].set_xlim(zcenter_show - 2.*tpack, zcenter_show + 2. * tpack)
    ax[1].set_ylim(n[(z > zcur1)&(z < zcur2)].min(), n[(z>zcur1)&(z<zcur2)].max())
    # cb = colorbar()
    # cb.set_label(r'$z$')
    xlabel(r'$z$')   ;   ylabel(r'$n_{\rm p}$') 
    title(r'$\omega_{\rm p} t = '+str(round(t, ndigits))+'$')
    ylim(maximum(n.min(),0.01), n.max())
    yscale('log')
    fig.set_size_inches(12.,6.)
    savefig(ddir + '/slewn{:05d}.png'.format(ctr))

    # close('all')
    
def slew_eplot(tlist, mlist, emelist, paelist, omega0, ddir = './'):
    
    clf()
    plot(tlist, mlist, 'k.')
    xlabel(r'$t$')  ;  ylabel(r'$M_{\rm tot}$')
    savefig(ddir + '/m.png')
    clf()
    plot(tlist, emelist, 'k.', label = 'EM')
    plot(tlist, paelist, 'rx', label = 'particles')
    plot(tlist, paelist+emelist, 'g--', label = 'total')
    ylim((paelist+emelist).max()*1e-5, (paelist+emelist).max()*2.)
    yscale('log')
    legend()
    xlabel(r'$t$')  ;  ylabel(r'$E$')
    savefig(ddir + '/e.png')
    
    clf()
    plot(tlist, tlist * 0. + 1. / omega0**2)
    plot(tlist, paelist/emelist)
    #    yscale('log')
    xlabel(r'$t$')  ;  ylabel(r'$E_{\rm PA}/E_{\rm EM}$')
    savefig(ddir + '/erat.png')
    
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
        plot(z1, (Aleft((z1-z0-t1+tmid)%zlen)*EyA/omega0)**2/2., 'b-', label = r'$A^2/2$')
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
    
    t1, z1, zhalf1, E1, B1, u1, n1, zcur1 = hio.fewout_readdump(hname1, nctr[0], ifzcur = True)
    Ex1, Ey1 = E1
    Bx1, By1 = B1
    ux1, uy1, uz1 = u1

    t2, z2, zhalf2, E2, B2, u2, n2, zcur2 = hio.fewout_readdump(hname2, nctr[1], ifzcur = True)
    Ex2, Ey2 = E2
    Bx2, By2 = B2
    ux2, uy2, uz2 = u2

    zlen = maximum(z1, z2).max() - minimum(z1, z2).min()
    zcenter = -zlen/2. + t1-tmid # used for the simulations without a source
    # zcenter = (zcenter - z0.min()) % zlen + z0.min()
    zcenter_show = minimum(maximum(zcenter, minimum(z1, z2).min()), maximum(z1, z2).max())
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
        ylim(-sqrt((Bx1**2).max()+(Bx2**2).max()), sqrt((Bx1**2).max()+(Bx2**2).max()))
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
        ylim(-sqrt((uy1**2).max()+(uy2**2).max()), sqrt((uy1**2).max()+(uy2**2).max()))
    legend()
    # title(r'$t = '+str(round(t1, ndigits))+'$')
    xlabel(r'$z$') ; ylabel(r'$u^{y, z}$')
    if zrange is not None:
        xlim(zrange[0], zrange[1])
    fig.set_size_inches(15.,5.)
    savefig('compareu{:05d}_'.format(nctr[0])+'{:05d}'.format(nctr[1]))
    close()
    clf()
    fig = figure()
    plot(z1, n1, 'k-', label = hname1)
    plot(z2, n2, 'r:', label = hname2, linewidth = 2)
    title(r'$t = '+str(round(t1, ndigits))+'$')
    xlabel(r'$z$') ; ylabel(r'$B_x$')
    if zrange is not None:
        xlim(zrange[0], zrange[1])
    fig.set_size_inches(12.,5.)
    savefig('comparen{:05d}_'.format(nctr[0])+'{:05d}'.format(nctr[1]))
    clf()
    fig, ax = subplots(ncols=2, figsize=(8, 4))
    ax[0].plot(z1, zcur1, 'k-', label = hname1)
    ax[0].plot(z2, zcur2, 'r:', label = hname2, linewidth = 2)
    fig.suptitle(r'$t = '+str(round(t1, ndigits))+'$')
    ax[0].set_xlabel(r'$z_0$') ; ax[1].set_xlabel(r'$z_0$') ; ax[0].set_ylabel(r'$z$')
    if zrange is not None:
        ax[1].set_xlim(zrange[0], zrange[1])
    else:
        ax[1].set_xlim(zcenter_show - 2.*tpack, zcenter_show + 2. * tpack)
    fig.set_size_inches(12.,5.)
    savefig('comparez{:05d}_'.format(nctr[0])+'{:05d}'.format(nctr[1]))
   
    
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
        
def energies(hname, narr, zmin = 0.):
    '''
    energies as functions of time, with a coordinate cut-off
    '''
    
    snarr = size(narr)

    if snarr <= 1:
        # reading a single file
        t, z0, zhalf, E, B, u, n, z = hio.fewout_readdump(hname, narr, ifzcur = True)
        Ex, Ey = E
        Bx, By = B
        ux, uy, uz = u

        EEM = simpson((Ex**2 + Ey**2)[z0>zmin], x = z0[z0>zmin])/2. + simpson((Bx**2 + By**2)[z0>zmin], x = z0[z0>zmin])/2.
        EEM0 = simpson(Ex**2 + Ey**2, x = z0)/2. + simpson(Bx**2 + By**2, x = z0)/2.

        gamma = sqrt(1. + ux**2+uy**2+uz**2)
        
        EPA0 = simpson((n * gamma * (gamma-1.)), x = z)
        EPA = simpson((n * gamma * (gamma-1.))[z0>zmin], x = z[z0>zmin])

        print(EEM0, EEM)
        print(EPA0, EPA)
        
        return t, (EEM0, EEM), (EPA0, EPA)
    else:

        t = zeros(snarr) ; eem = zeros(snarr) ; eem0 = zeros(snarr)
        epa = zeros(snarr) ; epa0 = zeros(snarr)
        
        for k in arange(snarr):
            ttmp, eemtmp, epatmp = energies(hname, narr[k], zmin = zmin)
            t[k] = ttmp
            eem0[k] = eemtmp[0] ; eem[k] = eemtmp[1]
            epa0[k] = epatmp[0] ; epa[k] = epatmp[1]

        clf()
        fig = figure()
        plot(t, epa0, 'r:', label= 'particles, total')
        plot(t, epa, 'r-', label= 'particles, z > '+str(zmin))
        plot(t, eem0, 'k:', label= 'electromagnetic, total')
        plot(t, eem, 'k-', label= 'electromagnetic, z > '+str(zmin))
        legend()
        
        xlabel(r'$t$') ;     ylabel(r'$E$')
        yscale('log') # ; xscale('log')
        ylim(eem.max()*1e-4, eem0.max()*1.5)
        savefig('eens.png')

        clf()
        fig = figure()
        plot(t, t*0. + 1./omega0**2 * (1.+(EyA/omega0)**2/8.), 'g--', r'$\omega^{-2}$')
        plot(t, epa0/eem0, 'r:', label= 'particles/EM, total')
        plot(t, epa/eem, 'k-', label= 'particles/EM, z > '+str(zmin))
        legend()
        
        xlabel(r'$t$') ;     ylabel(r'$E_{\rm PA}/E_{\rm EM}$')
        yscale('log') # ; xscale('log')
        ylim(1e-4, 1.2)
        savefig('eensrat.png')

def energy_compare(dir1, dir2):
    
    lines1 = loadtxt(dir1+'/slew_energy.dat')
    t1 = lines1[:,0] ;  m1 = lines1[:, 1] ; eem1 = lines1[:, 2] ; epa1 = lines1[:, 3]
    lines2 = loadtxt(dir2+'/slew_energy.dat')
    t2 = lines2[:,0] ;  m2 = lines2[:, 1] ; eem2 = lines2[:, 2] ; epa2 = lines2[:, 3]

    show()
    close('all')
    clf()
    fig, ax = subplots(ncols = 1, nrows=2, figsize=(4, 8))
    ax[0].plot(t1, eem1, 'k--', label = dir1+': EM')
    ax[0].plot(t2, eem2, 'r--', label = dir2+': EM')
    #    ax[0].plot(t1, epa1, 'k:', label = dir1+': particles')
    #    ax[0].plot(t2, epa2, 'r:', label = dir2+': particles')
    ax[0].plot(t1, eem1+epa1, 'k-', label = dir1+': total')
    ax[0].plot(t2, eem2+epa2, 'r-', label = dir2+': total')
    emax = maximum((eem1+epa1).max(), (eem2+epa2).max())
    ax[0].set_ylim(emax - maximum(epa1.max(), epa2.max())*3.0, emax+maximum(epa1.max(),epa2.max())*0.5)
    # yscale('log')
    ax[1].plot(t1, epa1, 'k:', label = dir1+': particles')
    ax[1].plot(t2, epa2, 'r:', label = dir2+': particles')    
    ax[0].set_xlabel(r'$t$') ;     ax[0].set_ylabel(r'$E$')
    ax[1].set_xlabel(r'$t$')  ;     ax[1].set_ylabel(r'$E$')
    ax[0].legend()
    ax[0].set_title('EM energy')
    ax[1].set_title('particle energy')
    fig.tight_layout()
    fig.savefig('energy_compare.png')
    show()

def test_monotonic_split():

    x = arange(1000)*0.01
    
    y = (x-1.)*(x-2.)*(x-3.)

    ylist = utile.monotonic_split(y)

    slist = len(ylist)

    print("number of monotonic regions = ", slist)
    
    clf()
    for k in arange(slist):
        print(x[ylist[k]].min(), '..', x[ylist[k]].max())
        plot(x[ylist[k]], y[ylist[k]], label = str(k))
    legend()
    xlabel(r'$x$') ;  ylabel(r'$y$')
    show()

def split_exception(x, y):

    ylist = utile.monotonic_split(y)
    slist = len(ylist)

    print("exception: number of monotonic regions = ", slist)

    xmax = x.min() ; xmin = x.max()
    ymax = y.min() ; ymin = y.max()
    
    clf()
    for k in arange(slist):
        print(x[ylist[k]].min(), '..', x[ylist[k]].max())
        plot(x[ylist[k]], y[ylist[k]], label = str(k))
        if k < (slist-1):
            xmax = maximum(xmax, x[ylist[k]].max())
            xmin = minimum(xmin, x[ylist[k]].min())
            ymax = maximum(ymax, y[ylist[k]].max())
            ymin = minimum(ymin, y[ylist[k]].min())
    xlim(xmin, xmax)  ; ylim(ymin, ymax)
    legend()
    xlabel(r'$x$') ;  ylabel(r'$y$')
    show()
   
    
