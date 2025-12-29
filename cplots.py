import matplotlib
from matplotlib.pyplot import *
from numpy import *

from scipy.integrate import simpson

ndigits = 2

def avec(t, z):
   return cos(2.*pi*(t-z)) # * (cos(pi*(t-z)/15.))**12.

def abs_trapezoid(y, x=None):
   return (abs(x[1:]-x[:-1])*(y[1:]+y[:-1])).sum()/2.

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

def ascmesh(narr, ddir = "A30B100nofeed"):

    nnarr = size(narr)
    
    t = zeros(nnarr)
    uylist = []
    uzlist = []
    zlist = []

    clf()
    for k in arange(nnarr):
        filename = ddir+"/asc"+str(narr[k])+".dat"
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
   savefig(ddir + '/zuz{:05d}.png'.format(nfile))
   
   if ifshow:
      show()
   else:
      close()

def gammas(nfile, ddir = 'A30B100', ifshow = False):

   filename = ddir + "/asc"+str(nfile)+".dat"
   print(filename)
   lines = loadtxt(filename)
   t = lines[0,0]
   z = lines[:,1]
   uy = lines[:,2]
   uz = lines[:,3]
   
   gamma = sqrt(1.+uz**2+uy**2)
   
   if ifshow:
      clf()
      plot(z, gamma, 'k.')
      xlabel(r'$z$')  ;  ylabel(r'$\gamma$')
      show()
      
   return t, z, gamma, uz

def allgammas(narr,  ddir = 'A30B100',  a = 30., b = 100., omega = 10.):
   
   nnarr = size(narr)

   tar = zeros(nnarr)  ;  gmean = zeros(nnarr) ; gstd = zeros(nnarr)
   uzmean = zeros(nnarr) # ; uzstd = zeros(nnarr)
   
   for k in arange(nnarr):
      ttmp, ztmp, gtmp, utmp = gammas(narr[k], ddir = ddir)
      tar[k] = ttmp ;  gmean[k] = abs_trapezoid(gtmp, x = ztmp)/abs_trapezoid(gtmp*0.+1., x = ztmp)
      gstd[k] = sqrt(abs_trapezoid(gtmp**2, x = ztmp)/abs_trapezoid(gtmp*0.+1., x = ztmp)-gmean[k]**2)
      uzmean[k] = abs_trapezoid(utmp, x = ztmp)/abs_trapezoid(utmp*0.+1., x = ztmp)
      # utmp.mean() ; uzstd[k] = utmp.std()
   
   # a = 30.; b = 100.; omega = 10.
   gamma_sobacchi = ( 7./3. * a**2 * b**(1./3.) * omega**(2./3.) * tar)**(3./7.)
   
   clf()
   errorbar(tar, gmean, yerr = gstd, fmt = 'ok', mfc = 'none', label = r'$\langle \gamma\rangle$')
   # errorbar(tar, sqrt(uzmean**2+1.), yerr = (sqrt(uzmean**2+uzstd**2+1.)-sqrt(maximum(uzmean**2-uzstd**2,0.)+1.))/2., fmt = 'xb', mfc = 'none', linewidth=2, label=r'$\gamma_{\rm bulk}$')
   plot(tar, sqrt(uzmean**2+1.), 'xb', label=r'$\gamma_{\rm bulk}$')
   plot(tar, gamma_sobacchi/2., 'r--', label='Sobacchi (2025)')
   plot(tar, tar*0.+1.+a**2/4., 'g:', label=r'$1.+a^2/4$')
   legend()
   xlabel(r'$\omega t / 2\pi$')  ;  ylabel(r'$\gamma$')
   savefig(ddir+'/allgammas.png')
   show()

   # mean uz
   #   clf()
   #   errorbar(tar, uzmean, yerr = uzstd, fmt = 'ok', mfc = 'none')
   #   xlabel(r'$\omega z / 2\pi$')  ;  ylabel(r'$\gamma$')
   #   savefig(ddir+'/alluzs.png')  
   #   show()
 
def zuzcompare(nfile, ddirs = ['A30B100nofeed', 'A30B100'], ifshow = False, zrange = None):

    filename1 = ddirs[0] + "/asc"+str(nfile)+".dat"
    lines = loadtxt(filename1)
    t1 = lines[0,0]
    z1 = lines[:,1]
    uy1 = lines[:,2]
    uz1 = lines[:,3]

    filename2 = ddirs[1] + "/asc"+str(nfile)+".dat"
    lines = loadtxt(filename2)
    t2 = lines[0,0]
    z2 = lines[:,1]
    uy2 = lines[:,2]
    uz2 = lines[:,3]

    print(filename1, filename2)

    clf()
    plot(z1, uz1, 'k-', label = ddirs[0])
    plot(z2, uz2, 'r:', label = ddirs[1], linewidth = 2)
    legend()
    if zrange is not None:
        xlim(zrange[0], zrange[1])
    xlabel(r'$\omega z/2\pi$') ; ylabel(r'$u^z$')
    title(r'$\omega t/2\pi = '+str(round(t1, ndigits))+'$')
    savefig('comzuz{:05d}.png'.format(nfile))

    if ifshow:
        show()
    else:
        close()

def fieldplot(nfile, zrange = None, ddir = 'A30B0', ifshow = False, acoeff = 30., omega  = 10.):

    filename = ddir + "/ascmax"+str(nfile)+".dat"
    print(filename)
    lines = loadtxt(filename)
    t = lines[0,0]
    z = lines[:,1]
    ear = lines[:,2]
    bar = lines[:,3]

    clf()
    # fig, ax = subplots(ncols=2, figsize=(12, 6))
    plot(z, ear, 'b--', label = r'$E$')
    plot(z, bar, 'r:', label = r'$B$')
    xlabel(r'$z$')
    legend()
    savefig('earbar{:05d}.png'.format(nfile))
    if ifshow:
       show()
    
def GOcompare(nfile, zrange = None, ddir = 'A30B0', ifshow = False, acoeff = 30., omega  = 10.):
    # compare non-magnetized solution to GO

    filename = ddir + "/asc"+str(nfile)+".dat"
    print(filename)
    lines = loadtxt(filename)
    t = lines[0,0]
    z = lines[:,1]
    uy = lines[:,2]
    uz = lines[:,3]

    # vector potential:
    a = avec(t, z) * acoeff
    
    clf()
    plot(a, a**2/2., 'r-')
    plot(uy, uz, 'k.')
    ylabel(r'$u^z$') ; xlabel(r'$u^y$')
    savefig(ddir + '/GOyz{:05d}.png'.format(nfile))

    gamma = sqrt(1.+uz**2+uy**2)
    
    print(uz.mean(), uz.std())
    print(trapezoid(uz, x = z)/trapezoid(uz*0.+1., x = z))
    
    zs = argsort(z)
    
    clf()
    fig, ax = subplots(ncols=2, figsize=(12, 6))
    ax[1].plot(z, uz, 'k.')
    ax[1].plot(z[zs], a[zs]**2/2., 'r-')
    ax[0].plot(z, uy, 'k.')
    ax[0].plot(z[zs], a[zs], 'r-')
    if zrange is not None:
        ax[0].set_xlim(zrange[0], zrange[1])
        ax[1].set_xlim(zrange[0], zrange[1])
    ax[0].set_xlabel(r'$\omega z/2\pi$') ; ax[1].set_ylabel(r'$u^z$')
    ax[0].set_ylabel(r'$u^y$')
    #ax[0].set_title(r'$u^y$')
    # ax[1].set_title(r'$u^z$')
    suptitle(r'$\omega t/2\pi = '+str(round(t, ndigits))+'$')
    savefig(ddir + '/GO{:05d}.png'.format(nfile))
    
    if ifshow:
        show()
    else:
        close()
    
        
def Cenergies(nfile, ddir = "A30B100"):
    
    u_filename = ddir + "/asc"+str(nfile)+".dat"
    lines = loadtxt(u_filename)
    t = lines[0,0]
    z = lines[:,1]
    uy = lines[:,2]
    uz = lines[:,3]
    gamma = sqrt(1.+uy**2+uz**2)

    EB_filename = ddir + "/ascmax"+str(nfile)+".dat"
    lines = loadtxt(EB_filename)
    t = lines[0,0]
    z0 = lines[:,1]
    e = lines[:,2]
    b = lines[:,3]

    Epa = trapezoid((gamma-1.), x = z0)
    Ema = trapezoid(e**2+b**2, x = z0)/2.

    return t, Epa, Ema

def Cenergy_plot(narr, ddir = "A30B100"):

    nnarr = size(narr)
    
    tar = zeros(nnarr) ;  epar = zeros(nnarr) ; emar = zeros(nnarr)

    for k in arange(nnarr):
        ttmp, Epatmp, Ematmp = Cenergies(narr[k], ddir = ddir)
        tar[k] = ttmp ;  epar[k] = Epatmp ; emar[k] = Ematmp
        print(narr[k])
        
    clf()
    plot(tar, epar, 'k.', label = 'particles')
    plot(tar, emar, 'rx', label = 'energies')
    plot(tar, emar+epar, 'b--', label = 'total')
    legend()
    xscale('log')  ; yscale('log')
    ylim(epar.max()/2., (emar+epar).max()*1.2)
    savefig('Cenergy.png')
    show()

#for k in arange(2000):
#      GOcompare(k,ddir="A10B0nofeed", acoeff=10.0)
#      GOcompare(k,ddir="A10B0", acoeff=10.0)
