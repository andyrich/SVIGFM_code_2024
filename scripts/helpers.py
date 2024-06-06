import flopy
from flopy.discretization.structuredgrid import StructuredGrid
import numpy as np
import pyemu

def get_sr():
    delr, delc = np.ones((85)) * 500, np.ones((275)) * 500
    xul = 6382956.489134505
    yul = 1918132.341874674
    angrot = 23.
    xll, yll = offset(xul, yul, delr, delc, angrot=angrot)
    
    m = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(m)
    
    m.modelgrid.set_coord_info(xoff=xul, yoff=yul, epsg=2226, angrot=angrot, )
    
    sr = pyemu.helpers.SpatialReference(delr=delr, delc=delc, xul=xul, yul=yul, units='feet',
                        proj4_str='EPSG:2226', rotation=23 , lenuni = 1, length_multiplier = 1.0 )

    return sr
    

def offset(xul, yul, delr, delc, angrot=0):
    '''
    convert upper left x and y to lower lower left x/y for model grid creatcion
    :param xul:
    :param yul:
    :param delr:
    :param delc:
    :param angrot: in degreess from topleft
    :return: xnew, ynew
    '''
    y = np.sum(delc)
    yoff = y * np.cos(angrot * np.pi / 180.)

    xoff = y * np.sin(angrot * np.pi / 180.)

    print(yoff, xoff)
    xnew = xul + xoff
    ynew = yul - yoff

    return xnew, ynew