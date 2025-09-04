import h5py
import os.path
from numpy import arange, size
from scipy.interpolate import interp1d

def entryname(n, ndig = 6):
    entry = str(n).rjust(ndig, '0') # allows for 6 positions (hundreds of thousand of entries)
    return entry

def okplane_hout(nu, k, c, hname = 'c.hdf', dataname = 'C'):
    # temporal frequency, spacial frequency, 2D complex amplitudes

    hfile = h5py.File(hname, "w")
    
    # TODO: there should be globals here
    #glo = hfile.create_group("globals")
    grp = hfile.create_group("nukeplane")
    grp.create_dataset(dataname, data=c, dtype='complex')
    grp.create_dataset('freq', data=nu)
    grp.create_dataset('wavenumber', data=k)

    hfile.flush()
    hfile.close()
    
def okplane_hread(hname, datanames = []):

    hfile = h5py.File(hname, 'r', libver='latest')

    nukeplane = hfile["nukeplane"]

    nu = nukeplane["freq"][:]
    k = nukeplane["wavenumber"][:]

    datalist = []
    nd = len(datanames)

    if nd > 0:
        
        for j in arange(nd):
            datalist.append(nukeplane[datanames[j]])

    return nu, k, datalist
