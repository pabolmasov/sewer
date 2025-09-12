import matplotlib
from matplotlib.pyplot import *

# HDF5 io:
import hio

from numpy import *

cmap = 'viridis'

def show_nukeplane(omega0 = 1.0, bgdfield = 0.):

    omega, k, datalist = hio.okplane_hread('okplane_Bx.hdf', datanames = ['Bx'])
    bxlist_FF =  datalist[0][:,:]
    print(type(bxlist_FF[0,0]))
    print(type(omega[0]))
    print(type(k[0]))
    
    babs = sqrt(bxlist_FF.real**2 + bxlist_FF.imag**2)
    # fftshift(nu)
    # fftshift(k)
    babs = log10(ma.masked_array(babs, mask = (babs <= 0.)))
    
    clf()
    fig = figure()
    pcolormesh(omega, k, transpose(babs), vmin = maximum(babs.min(), babs.max()-3.), vmax = babs.max())

    cb = colorbar()
    cb.set_label(r'$\log_{10} |\tilde{b}_x|$')
    #plot([-f0], [ f0], 'ro', mfc='none')
    #plot([f0], [ -f0], 'ro', mfc='none')

    ktmp = 3. * omega0 * (arange(100)/double(100)-0.5)
    
    if abs(bgdfield)>0.01:
        # circularly polarized components
        plot(ktmp, (ktmp - 1./(ktmp+bbgdz)), 'b:')
        plot(ktmp, (ktmp - 1./(ktmp-bbgdz)), 'r:')        
        plot(ktmp, -(ktmp - 1./(ktmp+bbgdz)), 'r:')
        plot(ktmp, -(ktmp - 1./(ktmp-bbgdz)), 'b:')        
    else:
        plot(sqrt(1.+ktmp**2), ktmp, 'w:')
        plot(-sqrt(1.+ktmp**2), ktmp, 'w:')
        plot(sqrt(1.+ktmp**2), -ktmp, 'w:')
        plot(-sqrt(1.+ktmp**2), -ktmp, 'w:')
    
    xlim(-2. * omega0 , 2. * omega0 )
    ylim(-2. * omega0, 2. * omega0)
    # xlim(omega.min(), omega.max())
    # ylim(k.min(), k.max())
    
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

def maps_dat(filename = "sewerout.dat"):

    lines = loadtxt(filename)

    t2 = lines[:,0]  ;   z2 = lines[:,1]  ;  q = lines[:,2]

    nt = size(unique(t2))   ; nz = size(unique(z2))

    t2 = reshape(t2, [nt, nz]) ;  z2 = reshape(z2, [nt, nz]) ; q = reshape(q, [nt, nz])

    clf()
    fig = figure()
    pcolormesh(z2, t2, q)
    colorbar()
    plot(z2[0,:], z2[0,:], 'w--')
    xlabel(r'$z$') ; ylabel(r'$t$')  
    ylim(t2.min(), t2.max())
    fig.set_size_inches(15.,10.)
    savefig('qmap.png')
    

def maps(z, tlist, bxlist, uzlist, nlist, ctr):
    s = shape(nlist)
    print(s)
    # if the two species are stored separately
    if s[0] == 2:
        nplist = nlist[0]
        nelist = nlist[1]
    s = shape(uzlist)
    print(s)
    # if the two species are stored separately
    if s[0] == 2:
        uzplist = uzlist[0]
        uzelist = uzlist[1]
    # 2D-visualization
    clf()
    fig = figure()
    pcolormesh(z, tlist, bxlist)
    colorbar()
    plot(z, z, 'w--')
    ylim(tlist.min(), tlist.max())
    xlabel(r'$z$') ; ylabel(r'$t$')  
    fig.set_size_inches(15.,10.)
    savefig('EBmap.png')
    
    clf()
    fig, ax = subplots(ncols=2, figsize=(8, 4))
    pc1 = ax[0].pcolormesh(z, tlist, uzplist)
    fig.colorbar(pc1, ax = ax[0])
    pc2 = ax[1].pcolormesh(z, tlist, uzelist)
    fig.colorbar(pc2, ax = ax[1])
    ax[0].set_xlabel(r'$z$') ; ax[1].set_xlabel(r'$z$') ; ax[0].set_ylabel(r'$t$')  
    savefig('uzmap.png'.format(ctr))
    
    clf()
    fig, ax = subplots(ncols=2, figsize=(8, 4))
    pc1 = ax[0].pcolormesh(z, tlist, nplist)
    fig.colorbar(pc1, ax = ax[0])
    pc2 = ax[1].pcolormesh(z, tlist, nelist)
    fig.colorbar(pc2, ax = ax[1])
    ax[0].set_xlabel(r'$z$') ; ax[1].set_xlabel(r'$z$') ; ax[0].set_ylabel(r'$t$')  
    savefig('nmap.png'.format(ctr))
    
    close()

def onthefly(z, zshift, ax0, ay0, az0, bx0, by0, ax, ay, az, bx, by, ux, uy, uz, n, ctr, t):

    s = shape(n)
    print(s)
    # if the two species are stored separately
    if s[0] == 2:
        np = n[0]
        ne = n[1]
    
    clf()
    fig, axes = subplots(5)
    # print(shape(z), shape(ax))
    axes[0].plot(zshift, ax0, 'r.')
    axes[0].plot(z, ax, 'k-')
    axes[1].plot(zshift, ay0, 'r.')
    axes[1].plot(z, ay, 'k-')
    axes[2].plot(zshift, az0, 'r.')
    axes[2].plot(z, az, 'k-')
    axes[3].plot(zshift, bx0, 'r.')
    axes[3].plot(z, bx, 'k-')
    axes[4].plot(zshift, by0, 'r.')
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
    if s[0] == 2:
        plot(z, np, 'k-', label = r'$n_+$')
        plot(z, ne, 'r:', label = r'$n_-$')
    else:
        plot(z, n, 'k-', label = r'$n$')
    xlabel(r'$z$')  ;   ylabel(r'$n$') 
    legend()
    title(r'$t = '+str(round(t))+'$')
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
