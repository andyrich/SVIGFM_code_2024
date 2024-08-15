import flopy
from flopy.discretization.structuredgrid import StructuredGrid
import numpy as np
import pyemu
from shapely.geometry import Point
from matplotlib.gridspec import GridSpec
from conda_scripts import sv_budget
import matplotlib.pyplot as plt



import os
import pandas as pd

import matplotlib.pyplot as plt
import conda_scripts.plot_help as ph

import matplotlib as mpl

import flopy
import geopandas as gpd

from pyemu.pst.pst_utils import (
    SFMT,
    IFMT,
    FFMT,
    pst_config,
    parse_tpl_file,
    try_process_output_file,
)


def modflow_hob_to_instruction_file(hob_file, ins_file=None):
    """write an instruction file for a modflow head observation file

    Args:
        hob_file (`str`): the path and name of the existing modflow hob file
        ins_file (`str`, optional): the name of the instruction file to write.
            If `None`, `hob_file` +".ins" is used.  Default is `None`.

    Returns:
        **pandas.DataFrame**: a dataFrame with control file observation information

    """

    hob_df = pd.read_csv(
        hob_file,
        sep=r"\s+",
        skiprows=1,
        header=None,
        names=["simval", "obsval", "obsnme", "date",'decimal_year'],
    )

    hob_df.loc[:, "obsnme"] = hob_df.obsnme.apply(str.lower)
    hob_df.loc[:, "ins_line"] = hob_df.obsnme.apply(lambda x: "l1 !{0:s}!".format(x))
    hob_df.loc[0, "ins_line"] = hob_df.loc[0, "ins_line"].replace("l1", "l2")

    if ins_file is None:
        ins_file = hob_file + ".ins"
    f_ins = open(ins_file, "w")
    f_ins.write("pif ~\n")
    f_ins.write(
        hob_df.loc[:, ["ins_line"]].to_string(
            col_space=0,
            columns=["ins_line"],
            header=False,
            index=False,
            formatters=[SFMT],
        )
        + "\n"
    )
    hob_df.loc[:, "weight"] = 1.0
    hob_df.loc[:, "obgnme"] = "obgnme"
    f_ins.close()
    return hob_df

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
    

def make_plot(name,x,y, i=None, j=None, geom = None):
    '''make a hydrograph plot'''
    
    fig = plt.figure(figsize = (10,6))

    gs1 = GridSpec(1, 2, left=0.05, right=0.48, wspace=0.05,width_ratios=[7, 2],)
    ax = fig.add_subplot(gs1[0, :-1])

    ax.set_title(f'{name}\n\nSVIGFM V2 Simulated Head')  
    ph.baseline(ax,hard = True,yearstart = 1965)

    
    ax.tick_params(labelbottom=True, labelleft=True)
    ax.set_ylabel('Hydraulic Head (feet)')
    ax.grid(True)
    ax3 = fig.add_subplot(gs1[-1, -1])
    ax3.tick_params(labelbottom=False, labelleft=False)
    sv_budget.sv_budget.sv_mod_map(simple = True,ax = ax3)
    annotations = [child for child in ax3.get_children() if isinstance(child, mpl.text.Text)]
    annotations[0].remove()
    ax3.legend().remove()

    if i is not None:
        pt = geom.query(f"i=={i} and j =={j}").geometry.centroid
    else:
        pt = gpd.GeoSeries(Point(x, y), crs = 2226)   
        
    pt.plot(ax = ax3,  markersize = 40,marker = '*', color = 'r') 
    return fig, ax
    
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