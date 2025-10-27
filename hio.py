import h5py
import os.path
from numpy import *
from scipy.interpolate import interp1d

def entryname(n, ndig = 6):
    entry = str(n).rjust(ndig, '0') # allows for 6 positions (hundreds of thousand of entries)
    return entry

def okplane_hout(omega, k, c, hname = 'c.hdf', dataname = 'C'):
    # temporal frequency, spacial frequency, 2D complex amplitudes

    hfile = h5py.File(hname, "w")
    
    # TODO: there should be globals here
    #glo = hfile.create_group("globals")
    grp = hfile.create_group("nukeplane")
    grp.create_dataset(dataname, data=c, dtype='complex')
    grp.create_dataset('ofreq', data=omega)
    grp.create_dataset('wavenumber', data=k)

    hfile.flush()
    hfile.close()
    
def okplane_hread(hname, datanames = []):

    hfile = h5py.File(hname, 'r', libver='latest')

    nukeplane = hfile["nukeplane"]

    omega = nukeplane["ofreq"][:]
    k = nukeplane["wavenumber"][:]

    datalist = []
    nd = len(datanames)

    if nd > 0:
        
        for j in arange(nd):
            datalist.append(nukeplane[datanames[j]])

    return omega, k, datalist

def fewout_init(hname, attrs, z, zhalf = None):
    '''
    opening a file where all the results of fewer will be stored
    attrs is a dictionary
    '''
    hfile = h5py.File(hname, 'w', libver='latest')
    glo = hfile.create_group("globals")

    for key, value in attrs.items():
        glo.attrs[key] = value

    geom = hfile.create_group("geometry")
    geom.create_dataset("z", data = z)
    if zhalf is not None:
        geom.create_dataset("zhalf", data = zhalf)

    return hfile

def fewout_dump(hfile, ctr, t, E, B, u, n):
    '''
    dump single snapshot record
    '''
    entry = entryname(ctr)
    grp = hfile.create_group("entry"+entry)

    Ex, Ey = E
    Bx, By = B
    ux, uy, uz = u
    
    grp.attrs["t"] = t
    
    grp.create_dataset("Ex", data= Ex)
    grp.create_dataset("Ey", data= Ey)

    grp.create_dataset("Bx", data= Bx)
    grp.create_dataset("By", data= By)

    grp.create_dataset("ux", data= ux)
    grp.create_dataset("uy", data= uy)
    grp.create_dataset("uz", data= uz)
    grp.create_dataset("n", data= n) # n gamma

    hfile.flush()
    print("HDF5 output, entry"+entry+"\n", flush=True)

def fewout_readdump(hname, ctr):

    hfile = h5py.File(hname, 'r', libver='latest')

    geom=hfile["geometry"]
    glo=hfile["globals"]
   
    z = geom["z"][:]
    zhalf = geom["zhalf"][:]

    entry = entryname(ctr)
    data=hfile["entry"+entry]
    t = data.attrs["t"]
    Ex = data["Ex"][:] ;  Ey = data["Ey"][:]
    Bx = data["Bx"][:] ;  By = data["By"][:]
    ux = data["ux"][:] ;  uy = data["uy"][:]  ; uz = data["uz"][:]
    n = data["n"]

    hfile.close()
    
    return t, z, zhalf, (Ex, Ey), (Bx, By), (ux, uy, uz), n

def fewout_readall(hname, qua = 'Bx', zalias = 2, talias = 2):

    hfile = h5py.File(hname, 'r', libver='latest')

    geom=hfile["geometry"]
    glo=hfile["globals"]

    z = geom["z"][::zalias]
    nz = size(z)

    # hfile.keys()
    entries = list(hfile.keys())[:-2]
    if talias > 1:
        entries = entries[::talias]
    nentries = size(entries)
    
    t = zeros(nentries) ; q = zeros([nz, nentries])
    ctr = 0
    
    for kent in entries:
        data=hfile[kent]
        t[ctr] = data.attrs["t"]
        q[:, ctr] = data[qua][::zalias]
        ctr += 1
        
    return z, t, q
