import flopy
from flopy.discretization.structuredgrid import StructuredGrid
import numpy as np
import pyemu
from shapely.geometry import Point
from matplotlib.gridspec import GridSpec
from conda_scripts import sv_budget
import matplotlib.pyplot as plt
import conda_scripts
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


def set_obs_to_zero_for_manually_selected_obs(pst):
    '''
    set observations in the pest obs file form the welldetails to zero-weight
    :param pst:
    :return: pst.observation_data
    '''

    main = os.path.join("HOB_Creation", "WellDetails_20240911.xlsx")
    info = pd.read_excel(main, sheet_name="FinalForHOB")
    info.loc[:, 'station'] = info.loc[:, 'station_name'].str.lower()

    obs = pst.observation_data.copy()

    gwle = obs.obsnme.str.contains('gwle') | obs.obsnme.str.contains('maj') | obs.obsnme.str.contains('ddown')
    hobs = obs.obsnme.str.contains('hds_')
    ren = lambda x: x.split("_date:")[0].split(":")[-1] if 'hds' in x or 'gwle' in x else ''
    station = obs.obsnme.apply(ren)

    print(f"number of observations before filtering {obs.query('weight>0').shape[0]}\n")
    # print(f"number of hobs stations {obs.loc[hobs].station.nunique()}")
    print(f"number of hobs stations before filtering  {obs.loc[hobs & obs.weight > 0].observationname.nunique()}")
    # set stations with HOBS zero weight
    c = station.isin(info.loc[info.HOBS_zero_weight == 1].station) & hobs
    obs.loc[c, 'weight'] = 0
    print(
        f"number of hobs stations after filtering  for HOBS_zero_weight {obs.loc[hobs & obs.weight > 0].observationname.nunique()}")
    print(f"number of observations after filtering for HOBS_zero_weight {obs.query('weight>0').shape[0]}\n")

    # print(obs.query("weight>0").shape[0])
    # set stations wight GWLE to zero weight
    print(
        f"number of gwle stations before filtering  for ALL_GWLE_zero_weight {obs.loc[gwle & obs.weight > 0].station.nunique()}")
    c = station.isin(info.loc[info.ALL_GWLE_zero_weight == 1].station) & gwle
    obs.loc[c, 'weight'] = 0
    print(
        f"number of gwle stations after filtering for ALL_GWLE_zero_weight {obs.loc[gwle & obs.weight > 0].station.nunique()}")
    print(f"number of observations after filtering for ALL_GWLE_zero_weight {obs.query('weight>0').shape[0]}\n")

    print(f"number of stations with HOBS_zero_weight in info: {info.HOBS_zero_weight.sum()} ")
    print(f"number of stations with ALL_GWLE_zero_weight in info: {info.ALL_GWLE_zero_weight.sum()} ")

    print("filtering GWLE before 2010")
    print(
        f"number of gwle stations before filtering for gwle<2010 {obs.loc[gwle & obs.weight > 0].shape[0]}")
    # set gwle before 2010 to zero:
    date = pd.to_datetime(obs.date)
    obs.loc[gwle & (date.dt.year < 2010), 'weight'] = 0
    print(
        f"number of gwle stations after filtering for gwle<2010 {obs.loc[gwle & obs.weight > 0].shape[0]}")

    return obs


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
    

def make_plot(name,x,y, i=None, j=None, geom = None, return_plot_ax = False):
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

    if return_plot_ax:
        return fig, ax, ax3
    else:
        return fig, ax

def get_zones(ml):
    from shapely import Polygon

    z = np.genfromtxt(os.path.join(ml.model_ws, 'model_arrays', 'zonation_3.csv'), delimiter = ' ')
    
    zotther = z.copy()
    zotther[zotther>8] = 0
    
    z = conda_scripts.arich_functions.array2rc(z,'zone').astype({'zone':int,'row':int,'column':int})
    
    aliases = {1: 'Bay', 2: 'EastSide', 3: 'SouthCent', 4: 'Kenwood', 5: 'VOM', 6: 'AguaCal',7:'WestSide',8:'CitySon',9:'Highlands'}
    z.loc[:,'name'] =z.loc[:,'zone'].replace(aliases)
    z = z.query("zone!=0").loc[:,['row','column','zone','name']]

    z.loc[:,'geometry'] = z.apply(lambda x: Polygon(ml.modelgrid.get_cell_vertices(x['row']-1, x['column']-1)), axis = 1)

    z = gpd.GeoDataFrame(z, geometry = 'geometry', crs = 2226)
    
    return z

def plot_zones(ml,zones = None, fig = None,ax = None, label = False):
    if zones is None:
        zones = get_zones(ml)
    if fig is None:
        fig = plt.figure(figsize = (6,8), dpi = 250)
    if ax is None:
        mm= conda_scripts.make_map.make_map('Zones')
        ax = mm.plotloc(fig, locname = 'SON_MOD')
        
    zones.dissolve('name').reset_index().plot('name',  ax = ax, alpha = .5)
    zones.dissolve('name').reset_index().exterior.plot( ax = ax, alpha = .5)

    if label:
        conda_scripts.plot_help.label_poly(zones.dissolve('name').reset_index(),ax, column = 'name')

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


def get_info_sheet(ml):
    '''combine the well details from M&A  with zones and GWLE station info. add geometry, x/y etc'''
    # main = r"C:\GSP\sv\model\update_2024\scripts\HOB_Creation\WellDetails_20240911.xlsx"
    main = os.path.join("HOB_Creation","WellDetails_20240911.xlsx")
    df = pd.read_excel(main, sheet_name="All_Final")
    print(df.shape)

    info = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Easting_x, df.Northing_y), crs=2226)
    info.loc[:, 'station'] = info.loc[:, 'station_name'].str.lower()

    stats = pd.read_csv(os.path.join('..', 'waterlevel', 'GWLE', 'gwle_station_info.csv'), index_col=0).loc[:,
            ['Station Name', 'Depth', 'num_meas']]
    stats.loc[:, 'station'] = stats.loc[:, 'Station Name'].str.lower()
    stats = stats.drop(columns='Station Name')
    stats.loc[:, 'GWLE'] = True

    info = pd.merge(info, stats, on='station')
    info.loc[:, 'GWLE'] = info.loc[:, 'GWLE'].fillna(False)

    info = info.drop(columns=['TOS_1', 'BOS_1',
                              'TOS_2', 'BOS_2', 'TOS_3', 'BOS_3', 'TOS_4', 'BOS_4', 'TOS_5', 'BOS_5',
                              'TOS_6', 'BOS_6', 'TOS_7', 'BOS_7', 'TOS_8', 'BOS_8', 'x_coordinate', 'y_coordinate',
                              'Coordinate_Source.1',
                              'Suggest Inclusion in future update?', 'Layer_Total ', 'Map_Label', 'Row',
                              'Col', 'Node', 'RowCol',
                              ], errors='raise')

    other_info = pd.read_csv(os.path.join('..', 'waterlevel', 'stats.csv'), index_col=0)
    other_info.loc[:, 'station'] = other_info.loc[:, 'station_name'].str.lower()
    other_info = gpd.GeoDataFrame(other_info,
                                  geometry=gpd.points_from_xy(other_info.station_longitude, other_info.station_latitude,
                                                              crs=4326), ).to_crs(2226)

    # other_info.loc[:,'Easting'] = other_info.geometry.x
    # other_info.loc[:,'Northing'] = other_info.geometry.y
    other_info = other_info.loc[~other_info.loc[:, 'station'].isin(info.loc[:, 'station'])]
    other_info = other_info.loc[:, ['station', 'geometry']]

    info = pd.concat([info, other_info])

    zones = get_zones(ml)
    zones = zones.drop(columns=['zone']).rename(columns={'name': 'zone'})
    # zones.loc[:,'station'] = zones.loc[:,'station_name'].str.lower()
    info = gpd.sjoin(info, zones.loc[:, ['zone', 'geometry']], how='left').drop(
        columns=['index_right', 'station_no', 'GRID_ID ', 'station_name', 'Easting_x', 'Northing_y'])
    # info = info.rename(columns = {'Include?':'HOBS'})
    info.loc[:, 'HOBS'] = info.loc[:, 'Include?'] == 1
    info = info.drop(columns='Include?')
    info.loc[:, 'num_meas'] = info.loc[:, 'num_meas'].fillna(0)
    info.loc[:, 'Depth'] = info.loc[:, 'Depth'].fillna('Unknown')

    c = info.RMP
    info = info.drop(columns='RMP')
    info.loc[:, 'RMP'] = c == 'Y'
    # Rearrange columns
    new_order = info.columns[-7:].tolist() + info.columns[:-7].tolist()
    info = info[new_order]

    info.loc[:, 'Easting'] = info.geometry.x
    info.loc[:, 'Northing'] = info.geometry.y

    return info
