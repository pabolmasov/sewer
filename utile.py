from numpy import *

def bufferfun(x):

    f = copy(x)
    
    w0 = (x<=-1.)
    w1 = (x>=1.)
    wmid = (x < 1.) & (x > -1.)

    if w0.sum()  > 1:
        f[w0] = 0.
    if w1.sum() > 1:
        f[w1] = 1.

    if wmid.sum() > 1:
        f[wmid] = 0.5 + 0.75 * x[wmid] - 0.25 * x[wmid]**3

    return f

def parcoeff(v):
    # coefficients of 2nd order polynomials fitting any function on an extended grid to arbitrary points within
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
