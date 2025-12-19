import matplotlib
from matplotlib.pyplot import *
from numpy import *

ndigits = 2

def zintplot():

    lines = loadtxt("zintout.dat")

    t = lines[:,0] ;  z = lines[:,1]

    xi = t - z

    uy = lines[:,2]  ;  uz = lines[:,3]
    ay = lines[:,4] ; az = lines[:,5]

    clf()
    # fig = figure()
    fig, ax = subplots(ncols=2, figsize=(12, 6))
    ax[0].plot(xi, uy, 'k.', label = r'$u^y$')
    # ax[0].plot(xi, ay, 'r-', label = r'$-A$')
    ax[1].plot(xi, uz, 'k.', label = r'$u^z$')
    # ax[1].plot(xi, az, 'r-', label = r'$A^2/2$')
    ax[0].legend() ;   ax[1].legend()
    ax[0].set_xlabel(r'$\omega \xi / (2\pi)$') ; ax[0].set_ylabel(r'$u^{y}$')
    ax[1].set_xlabel(r'$\omega\xi / (2\pi)$') ; ax[1].set_ylabel(r'$u^{z}$')
    savefig('zinttest.png')
    # show()

    gamma = sqrt(1.+uy**2+uz**2)

    xicut = 5.
    gammacut = gamma[xi < xicut].max()

    a = 30. ; b = 100. ; omega = 10.   
    gamma_sobacchi = ( 7./3. * a**2 * b**(1./3.) * omega**(2./3.) * t)**(3./7.)
    
    close('all')
    clf()
    fig, ax = subplots(ncols=2, figsize=(12, 6))
    ax[0].plot(xi, gamma, 'k-')
    ax[0].plot(xi, gamma_sobacchi, 'r-')
    ax[1].plot(xi, gamma, 'k-')
    ax[0].plot([0.,xicut], [0.,0.], 'g:')
    ax[0].plot([0.,xicut], [gammacut,gammacut], 'g:')
    ax[0].plot([0.,0.], [0., gammacut], 'g:')
    ax[0].plot([xicut,xicut], [0., gammacut], 'g:')
    ax[1].set_xlim(0., xicut)
    ax[1].set_ylim(0., gamma[xi < xicut].max())
    ax[0].set_xlabel(r'$\omega\xi / (2\pi)$')
    ax[1].set_xlabel(r'$\omega\xi / (2\pi)$')
    ax[0].set_ylabel(r'$\gamma$')
    ax[1].set_ylabel(r'$\gamma$')
    savefig('zinttest_gamma.png')

    clf()
    scatter(uy, uz, c = xi)
    cb = colorbar()
    cb.set_label(r'$\omega\xi / (2\pi)$')
    xlabel(r'$u^{y}$')  ;  ylabel(r'$u^{z}$')
    savefig('zinttest_uyz.png')
    show()

def ascmesh(narr):

    nnarr = size(narr)
    
    t = zeros(nnarr)
    uylist = []
    uzlist = []
    zlist = []

    clf()
    for k in arange(nnarr):
        filename = "asc"+str(narr[k])+".dat"
        print(filename)
        lines = loadtxt(filename)
        t[k] = lines[0,0]
        if k == 0:
            z0 = lines[:,1]
        uylist.append(lines[:,2])
        uzlist.append(lines[:,3])
        zlist.append(lines[:,1])
        
    uylist = asarray(uylist)
    uzlist = asarray(uzlist)

    z2, t2 = meshgrid(z0, t)

    gamma = sqrt(1.+uzlist**2+uylist**2)
    
    clf()
    scatter(zlist, gamma, c = t2, s=0.1)
    yscale('log')
    colorbar()
    xlabel(r'$z$')  ;  ylabel(r'$\gamma$')
    savefig('zbreak.png')
    
    clf()
    fig, ax = subplots(ncols=2, figsize=(12, 6))
    pc1 = ax[0].pcolormesh(z0, t, uylist)
    cb1 = fig.colorbar(pc1, ax = ax[0])
    pc2 = ax[1].pcolormesh(z0, t, uzlist)
    cb2 = fig.colorbar(pc1, ax = ax[1])
    ax[0].set_title(r'$u^y$')  ;  ax[1].set_title(r'$u^z$')
    ax[0].set_xlabel(r'$\omega z_0 / (2\pi)$') ; ax[1].set_xlabel(r'$\omega z_0 / (2\pi)$') ; ax[0].set_ylabel(r'$\omega_{\rm p} t$')  
    savefig('uyzmap.png')
    
    # gamma statistics:
    meangamma = gamma.mean(axis =1)
    stdgamma = gamma.std(axis =1)

    a = 30. ; b = 100. ; omega = 10.   
    gamma_sobacchi = ( 7./3. * a**2 * b**(1./3.) * omega**(2./3.) * t)**(3./7.)
    
    clf()
    plot(t, meangamma, 'k-')
    plot(t, meangamma+stdgamma, 'k:')
    plot(t, meangamma-stdgamma, 'k:')
    plot(t, gamma_sobacchi, 'r--')
    plot(t, gamma_sobacchi/2., 'r--')
    xlabel(r'$\omega t / (2\pi)$') ; ylabel(r'$\gamma$')
    savefig('gammastat.png')

def zuzplot(nfile, zrange = None, ddir = 'A30B100', ifshow = False):
    
    filename = ddir + "/asc"+str(nfile)+".dat"
    print(filename)
    lines = loadtxt(filename)
    t = lines[0,0]
    z = lines[:,1]
    uy = lines[:,2]
    uz = lines[:,3]

    clf()
    plot(z, uz, 'k-')
    if zrange is not None:
        xlim(zrange[0], zrange[1])
    xlabel(r'$\omega z/2\pi$') ; ylabel(r'$u^z$')
    title(r'$\omega t/2\pi = '+str(round(t, ndigits))+'$')
    savefig('zuz{:05d}.png'.format(nfile))

    if ifshow:
        show()
    else:
        close()

# for k in arange(500):
#     zuzplot(k, zrange = [0.,1.])
