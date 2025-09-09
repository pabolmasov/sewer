import matplotlib
from matplotlib.pyplot import *

from numpy import *

cmap = 'viridis'


def show_nukeplane(f0 = 1.0):

    nu, k, datalist = hio.okplane_hread('okplane_Bx.hdf', datanames = ['Bx'])
    bxlist_FF =  datalist[0][:,:]
    print(type(bxlist_FF[0,0]))
    print(type(nu[0]))
    print(type(k[0]))

    print(nu)
    
    babs = sqrt(bxlist_FF.real**2 + bxlist_FF.imag**2)
    # fftshift(nu)
    # fftshift(k)
    babs = log10(ma.masked_array(babs, mask = (babs <= 0.)))
    
    clf()
    fig = figure()
    # pcolormesh(tlist, f/2./pi, transpose(Fbxlist.real), shading='nearest', vmin = -10, vmax = 10)
    pcolormesh(nu, k, transpose(babs), vmin = maximum(babs.min(), babs.max()-3.), vmax = babs.max())
    # pcolormesh(ofreq[:nthalf], f[:nzhalf]/2./pi, transpose(babs)[:nzhalf, :nthalf], vmin = maximum(babs.min(), babs.max()-5.), vmax = babs.max(), shading='nearest')
    # pcolormesh(ofreq[nthalf:], f[:nzhalf]/2./pi, transpose(babs)[:nzhalf, nthalf:], vmin = maximum(babs.min(), babs.max()-5.), vmax = babs.max(), shading='nearest')
    # pcolormesh(ofreq[:nthalf], f[nzhalf:]/2./pi, transpose(babs)[nzhalf:, :nthalf], vmin = maximum(babs.min(), babs.max()-5.), vmax = babs.max(), shading='nearest')
    # pcolormesh(ofreq[nthalf:], f[nzhalf:]/2./pi, transpose(babs)[nzhalf:, nthalf:], vmin = maximum(babs.min(), babs.max()-5.), vmax = babs.max(), shading='nearest')
    cb = colorbar()
    cb.set_label(r'$\log_{10} |\tilde{b}_x|$')
    #plot([-f0], [ f0], 'ro', mfc='none')
    #plot([f0], [ -f0], 'ro', mfc='none')

    ktmp = 20.*f0 * (arange(100)/double(100)-0.5)

    #plot(ktmp/2./pi, ktmp/2./pi, 'w--')
    #plot(-ktmp/2./pi, ktmp/2./pi, 'w--')
    #plot(ktmp/2./pi, -ktmp/2./pi, 'w--')
    #plot(-ktmp/2./pi, -ktmp/2./pi, 'w--')    
    '''
    if abs(bbgdz)>0.01:
        # circularly polarized components
        plot(ktmp/2./pi, (ktmp - 1./(ktmp+bbgdz))/2./pi, 'b:')
        plot(ktmp/2./pi, (ktmp - 1./(ktmp-bbgdz))/2./pi, 'r:')        
        plot(ktmp/2./pi, -(ktmp - 1./(ktmp+bbgdz))/2./pi, 'r:')
        plot(ktmp/2./pi, -(ktmp - 1./(ktmp-bbgdz))/2./pi, 'b:')        
    else:
        plot(sqrt(1.+ktmp**2)/2./pi, ktmp/2./pi, 'w:')
        plot(-sqrt(1.+ktmp**2)/2./pi, ktmp/2./pi, 'w:')
        plot(sqrt(1.+ktmp**2)/2./pi, -ktmp/2./pi, 'w:')
        plot(-sqrt(1.+ktmp**2)/2./pi, -ktmp/2./pi, 'w:')
    '''
    
    xlim(-2. * f0 , 2. * f0 )
    ylim(-2. * f0, 2. * f0)
    
    #    xlim(1./tmax, 1./dtout)  ;  ylim(1./zlen, 1./dz)
    fig.set_size_inches(15.,10.)
    xlabel(r'$\nu$') ; ylabel(r'$k$')    
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

def maps(tlist, bxlist, uzlist, nlist):
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

def onthefly(z, zshift, ax0, ay0, az0, bx0, by0, ax, ay, az, bx, by, ux, uy, uz, n, ctr, t):
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
    plot(z, n, 'k-', label = r'$n$')
    xlabel(r'$z$')  ;   ylabel(r'$n$') 
    #        legend()
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
