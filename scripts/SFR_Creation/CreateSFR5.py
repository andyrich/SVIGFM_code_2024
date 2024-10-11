"""
Created on Mon Sep 16 14:07:12 2024

    @author: Andrew Rich, Sonoma. Adapted by pwickham
    Script Name: BuildSFR.py
    Author: Andrew Rich, Adapted by Patrick Wickham
    Email: pwickham@elmontgomery.com
    Created: September 2024
    Updated: October 2024
        Developed by 2024 Sonoma Valley Water District and edited by Montgomery and Associates Water Resource Consultants
        This work is distributed without warranty of any kind. 
        
        For questions about utilizing this script contact Patrick Wickham
        pwickham@elmontgomery.com
    
    OVERVIEW
    This script creates an SFR network using sfrmaker. A custom polyline shapefile input is required.
    It overlays the model grid with your line shapefile and creates an SFR package. 
    Here we use a custom polyline shapefile, though a typical NHD shapefile can also be provided. 
    
    THEORY
        https://doi-usgs.github.io/sfrmaker/latest/inputs.html#hydrography-data
        
    MECHANICS
        pandas, sfrmaker, geopandas
        A lot of the internal workings of sfrmaker relavent to this script can be see in the sfrmaker Lines.py file
        (should be under something like C: users/ [you]/program data/conda/site-packages/ sfrmaker )
        
    DEVELOPMENT NOTES
        ` hard-imported some functions and customized so that SFRmaker would actually use arbolate sum and conductivity. otherwise, it was defaulting to a value of '1.0' for everything
            arbolate sum function imported and used here manually for width1 and width2 prior to creating lines. despite being called custom, it is the same math. 
        `conductivity had a hard coded setting to 'strhc1' =1 in SFRmaker in the Lines.py from_dataframe function, so made a custom version that doesnt do that. 
        ` after several attempts to get sfrmaker to accept that my cooridnate system was originally in meters but reprojected in python, I just gave up and reprojected in GIS. 
            The issue had to do with internal checking of metadata, which is not renamed by python reprojection.
            This is avoiding unnessesary conversions that SFRmaker was doing, because it was assuming meters for units. 
        Swap out the top line of the .sfr file after creation to make sure that relavent files are printed. 
"""
#===================================USER INPUTS==============================
StreamsFile=r'G:\GIS-Tuc\Projects\9400_SonomaValley\2024SVIGFMUpdate\SFR\Expansion_202409\Streams_SFR_2024_20240924_ForSFRMaker_Update10_CRS2226.shp'
ModelPath=r'C:\Users\pwickham\Documents\GitHub\SVIGFM_model'

ID_Field='NHDPlusID'
Routing_Field='ToNHDPID'
LengthField='Length_km' # length of the line segment. provided in km so we convert to feet (see length conversion)
#'Width_1'=,  # NOT DEFINED. USING ARBOLATE SUM INSTEAD. its basically an either-or situation. 
#Width_2= # SFRMaker calcualtes widths from the arbolate sum and the length, if you don't manually provide widths
ArbolateField2= 'asum2' #Downstream arbolate sum (sum of total length above this segment. Found in NHD datasets. provided in km, so converting to meters
UpperElevationField='elevup' #elevation at uppermost reach from lidar. provided in m, converted here to ft
LowerElevationField='elevdn' #elevation at lowermost reach from lidar. provided in m, converted here to ft

ConductivityField='strhc1' # conductivity field. You don't need this for SFRMaker to run but it saves the trouble of assigning conductivity later. No conversions coded, leave it in model units. 
GeometryField='geometry' # this is pretty much always called 'geometry', so don't worry about it 
LengthUnits='feet' # This is your DESIRED units for SFRmaker. Apply conversions below. length units for 'length' and arbolate sum, if provided. Typically, these are in km so remember to convert to m or feet
HeightUnits='feet' # This is your DESIRED units for SFRmaker. Apply conversions below. height units for UpperElevationField and LowerElevationField
WidthUnits='feet' # This is your DESIRED units for SFRmaker. Conversions made as part of custom arbolate sum function
ElevationConversion= 3.28084 # input the multiplicative conversion for elevation based fields (here its meters to feet)
LengthConversion= 1000*3.28084# input the multiplicative conversion for length based fields (here its km to feet)
#==================================IMPORT PACKAGES========================
import geopandas as gpd
import numpy as np
#import pdb
import pandas as pd
import os
import sfrmaker
import flopy
from flopy.discretization.structuredgrid import StructuredGrid
#import numpy as np 
import pyproj # unused by script?
from pyproj import CRS
#import time
#from sfrmaker.units import convert_length_units, get_length_units
#from sfrmaker.lines import Lines
#import warnings
#from shapely.geometry import box
#from sfrmaker.reaches import consolidate_reach_conductances, interpolate_to_reaches, setup_reach_data
#from gisutils import df2shp, get_authority_crs, get_shapefile_crs
#from sfrmaker.routing import pick_toids, find_path, make_graph, renumber_segments
#from sfrmaker.checks import routing_is_circular, is_to_one
#from sfrmaker.gis import read_polygon_feature, get_bbox, get_crs
#from sfrmaker.grid import StructuredGrid
#from sfrmaker.nhdplus_utils import load_nhdplus_v2, get_prj_file, load_nhdplus_hr
#from sfrmaker.sfrdata import SFRData
#from sfrmaker.units import convert_length_units, get_length_units
#from sfrmaker.utils import (width_from_arbolate_sum, arbolate_sum)
#from sfrmaker.reaches import consolidate_reach_conductances, interpolate_to_reaches, setup_reach_data
#from sfrmaker.routing import get_previous_ids_in_subset
#from sfrmaker.checks import routing_is_circular, is_to_one
#from sfrmaker.routing import pick_toids, find_path, make_graph, renumber_segments
print(sfrmaker.__version__)


#==================================DEFINE FUNCTIONS========================
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

def get_model(workspace=r'C:\\GSP\\sv\\model\\SV_mod_V2\\master', # from Andrew Rich of Sonoma Valley Water
              sfr_path="sv_GSP.sfr",
              dis_name="sv_model_grid_6layers_GSP.dis",
              historical=True):
    '''
    load the sv model as a flopy instance
    :param workspace:
    :param zones_:
    :param cbc:
    :param sfr_path:
    :param read_pickle:
    :param SFR_basin:
    :return:
    '''

    # ml = flopy.modflow.Modflow(model_ws =r'C:\GSP\sv\model\SV_model_lith_v9\lith_v9')
    ml = flopy.modflow.Modflow(model_ws=workspace)
    ml.change_model_ws(workspace)

    # load sfr file, but skip segment data inflows
    sfr_file = os.path.join(workspace, sfr_path)


    sfr_mod = flopy.modflow.ModflowSfr2.load(sfr_file, ml) # unused by script

    # dis = flopy.modflow.ModflowDis.load("C:\GSP\sv\model\SV_model_lith_v9\lith_v9\sv_model_grid_6layers.dis",ml)
    dis = flopy.modflow.ModflowDis.load(os.path.join(workspace, dis_name), ml) # unused by script

    zones = flopy.modflow.ModflowZon.load(os.path.join(workspace, 'sv_zones.zone'), ml) # unused by script

    # flopy.modflow.ModflowZon(ml,zones)
    # upw = flopy.modflow.ModflowUpw.load(os.path.join(workspace, 'sv_pp_plus_zones.upw'), ml)

    if  os.path.exists(os.path.join(workspace,'model_arrays', "ibound_1.txt")):
        # load dis file, but skip options
        ibnd = np.concatenate([np.expand_dims(np.genfromtxt(os.path.join(workspace,'model_arrays', f"ibound_{i+1}.txt"),
                                              dtype=int),axis  = 0)
                               for i in range(6)])
        print("Loaded ibound from arrays inside model_arrays")
    else:
        gdb = os.path.join(r'C:\GSP\sv\GIS', 'sv_model_geodatabase.gdb')
        active_grid = gpd.read_file(gdb, layer='active_grid_cells')
        print(f'loading ibound from {gdb}')

        ibnd = af.rw2aray(ml.dis.nrow, ml.dis.ncol, active_grid.row, active_grid.column_, # af? not called prior. 
                          np.ones(active_grid.shape[0], dtype=int), 'yo')
        ibnd = np.expand_dims(ibnd['yo'], 0)
        ibnd = np.concatenate((ibnd, ibnd, ibnd, ibnd, ibnd, ibnd), axis=0)

    bas = flopy.modflow.ModflowBas(ml, ibound=ibnd) # What is this for? Its in the function so you can't view it. Maybe its different in jupyter
    return ml

def Custom_width_from_arbolate_sum(asum, a=0.1193, b=0.5032, minimum_width=1., input_units='feet',  #same math as original width from arbolate sum function but just pulled out here because SFRmaker wasn't assigning properly in internal functions
                            output_units='feet'):
    """Estimate stream width from arbolate sum, using a power law regression
    comparing measured channel widths to arbolate sum.
    (after Leaf, 2020 and Feinstein et al. 2010, Appendix 2, p 266.)

    .. math::
        width = unit\_conversion * a * {asum_{(meters)}}^b

    Parameters
    ----------
    asum: float or 1D array
        Arbolate sum in the input units.
    a : float
        Multiplier parameter. Literature values:
        Feinstein et al (2010; Lake MI Basin): 0.1193
        Leaf (2020; Mississippi Embayment): 0.0592
    b : float
        Exponent in power law relationship. Literature values:
        Feinstein et al (2010; Lake MI Basin): 0.5032
        Leaf (2020; Mississippi Embayment): 0.5127
    minimum_width : float
        Minimum width to be returned. By default, 1.
    input_units : str, any length unit; e.g. {'m', 'meters', 'km', etc.}
        Length unit of asum
    output_units : str, any length unit; e.g. {'m', 'meters', 'ft', etc.}
        Length unit of output width

    Returns
    -------
    width: float
        Estimated width in feet

    Notes
    -----
    The original relationship described by Feinstein et al (2010) was for arbolate sum in meters
    and output widths in feet. The :math:`u` values above reflect this unit conversion. Therefore, the
    :math:`unit\_conversion` parameter above includes conversion of the input to meters, and output
    from feet to the specified output units.

    NaN arbolate sums are filled with the specified ``minimum_width``.

    References
    ----------
    see :doc:`References Cited <../references>`

    Examples
    --------
    Original equation from Feinstein et al (2010), for arbolate sum of 1,000 km:
    >>> width = width_from_arbolate_sum(1000, 0.1193, 0.5032, input_units='kilometers', output_units='feet')
    >>> round(width, 2)
    124.69
    """
    if not np.isscalar(asum):
        asum = np.atleast_1d(np.squeeze(asum))
    input_unit_conversion = convert_length_units(input_units, 'meters')
    output_unit_conversion = convert_length_units('feet', output_units)
    w = output_unit_conversion * a * (asum * input_unit_conversion) ** b
    if not np.isscalar(asum):
        w[w < minimum_width] = float(minimum_width)
        w[np.isnan(w)] = float(minimum_width)
    elif np.isnan(w) or w < minimum_width:
        w = minimum_width
    else:
        pass
    return w


def Custom_from_dataframe(cls, df,  # identical to original function but takes hydraulic conductivity as input rather than overwriting it with '1.0' which was what was being done prior. 
                       id_column='id',
                       routing_column='toid',
                       arbolate_sum_column2='asum2',
                       asum_units='km',
                       width1_column='width1',
                       width2_column='width2',
                       width_units='meters',
                       up_elevation_column='elevup',
                       dn_elevation_column='elevdn',
                       elevation_units='meters',
                       hydraulic_conductivity_column=None,  # Add the conductivity column as an optional argument
                       geometry_column='geometry',
                       name_column='name',
                       crs=None, prjfile=None,
                       **kwargs):
        """[summary]

        Parameters
        ----------
        df : DataFrame
            Pandas DataFrame or Geopandas GeoDataFrame
            with flowline information, including
            shapely :class:`LineStrings <LineString>` in a `'geometry'` column.
        id_column : str, optional
            Attribute field with line identifiers, 
            by default 'id'
        routing_column : str, optional
            Attribute field with downstream routing connections,
            by default 'toid'
        arbolate_sum_column2 : str, optional
            Attribute field with arbolate sums at downstream ends of lines, 
            by default 'asum2'
        asum_units : str, optional
            Length units for values in ``arbolate_sum_column2``; 
            by default 'km'.
        width1_column : str, optional
            Attribute field with channel widths at upstream ends of lines,
            by default 'width1'
        width2_column : str, optional
            Attribute field with channel widths at downstream ends of lines, 
            by default 'width2'
        width_units : str, optional
            Length units for values in ``width1_column`` and ``width2_column``; 
            by default 'meters'.
        up_elevation_column : str, optional
            Attribute field with elevations at upstream ends of lines, 
            by default 'elevup'
        dn_elevation_column : str, optional
            Attribute field with elevations at downstream ends of lines,
            by default 'elevdn'
        elevation_units : str, optional
            Length units for values in ``up_elevation_column`` and ``dn_elevation_column``; 
            by default 'meters'.
        name_column : str, optional
            Attribute field with feature names, 
            by default 'name'
        crs : obj, optional
            Coordinate reference object for ``df``. This argument is only needed
            if ``df`` is not a GeoDataFrame with a valid attached coordinate reference.
            Can be any of:
            - PROJ string
            - Dictionary of PROJ parameters
            - PROJ keyword arguments for parameters
            - JSON string with PROJ parameters
            - CRS WKT string
            - An authority string [i.e. 'epsg:4326']
            - An EPSG integer code [i.e. 4326]
            - A tuple of ("auth_name": "auth_code") [i.e ('epsg', '4326')]
            - An object with a `to_wkt` method.
            - A :class:`pyproj.crs.CRS` class
        prjfile: str, optional
            ESRI-style projection file with coordinate reference information for ``df``. 
            This argument is only needed if ``df`` is not a GeoDataFrame 
            with a valid attached coordinate reference.
        **kwargs : dict, optional
            Support for deprecated keyword options.

            .. deprecated:: 0.13
                The following arguments will be removed in SFRmaker 0.13.

                - ``attr_length_units`` (str): use ``width_units`` or ``asum_units`` instead.
                - ``attr_height_units`` (str): use ``elevation_units`` instead.
                - ``epsg`` (int): use ``crs`` instead.
                - ``proj_str`` (str): use ``crs`` instead.
             
        Returns
        -------
        lines : :class:`Lines` instance
        """

        assert geometry_column in df.columns, \
            "No feature geometries found: dataframe column '{}' doesn't exist.".format(geometry_column)
        assert routing_column in df.columns, \
            "No routing information found; dataframe column '{}' doesn't exist.".format(routing_column)

        # rename the columns for consistency
        rename_cols = {id_column: 'id',
                       routing_column: 'toid',
                       arbolate_sum_column2: 'asum2',
                       width1_column: 'width1',
                       width2_column: 'width2',
                       up_elevation_column: 'elevup',
                       dn_elevation_column: 'elevdn',
                       name_column: 'name'}

        # dictionary for assigning new column names
        rename_cols = {k: v for k, v in rename_cols.items() if k in df.columns and v != k}
        # drop any existing columns that have one of the new names
        # (otherwise pandas will create a DataFrame
        # instead of a Series under that column name)
        to_drop = set(rename_cols.values()).intersection(df.columns)
        df.drop(to_drop, axis=1, inplace=True)
        df.rename(columns=rename_cols, inplace=True)
        # Ensure 'strhc1' or hydraulic conductivity column is retained
        if hydraulic_conductivity_column is not None and hydraulic_conductivity_column in df.columns:
            # Include the conductivity column in the final dataframe
            df['strhc1'] = df[hydraulic_conductivity_column]
        # Organize columns (ensure 'strhc1' is included)
        column_order = ['id', 'toid', 'asum1', 'asum2', 'width1', 'width2', 'elevup', 'elevdn', 'name', 'geometry', 'strhc1'] # added strhc1 
        for c in column_order:
            if c not in df.columns:
                df[c] = 0
            else:
                assert isinstance(df[c], pd.Series)
        df = df[column_order].copy()

        return cls(df, 
                   asum_units=asum_units,
                   width_units=width_units, 
                   elevation_units=elevation_units,
                   crs=crs, prjfile=prjfile, **kwargs)
import types

sfrmaker.Lines.Custom_from_dataframe = types.MethodType(Custom_from_dataframe, sfrmaker.Lines) # assign custom function to class
#Lines.geometry_length_units = property(custom_geometry_length_units) # no longer used with manual gis reprojection


#==================================LOAD AND TRANSFORM MODEL========================

# Create the MODFLOW model object
m = flopy.modflow.Modflow()

# Define the model grid with 85 columns and 275 rows, cell size 500 feet
delr, delc = np.ones((85)) * 500, np.ones((275)) * 500  # Cell size in feet
xul = 6382956.489134505
yul = 1918132.341874674
angrot = 23.

# Calculate lower-left corner
xll, yll = offset(xul, yul, delr, delc, angrot=angrot)

# Set the MODFLOW discretization with length units in feet (lenuni=1)
dis = flopy.modflow.ModflowDis(m, nlay=1, nrow=275, ncol=85, delr=delr, delc=delc, 
                               top=0, botm=-100, lenuni=1)  # lenuni=1 sets length units to feet

# Set model grid information including coordinate system and length units
m.modelgrid.set_coord_info(xoff=xul, yoff=yul, epsg=2226, angrot=angrot)

# Create the structured grid with length units in feet
mg = StructuredGrid(delc=delc, delr=delr, xoff=xll, yoff=yll, angrot=angrot,
                    epsg=2226, lenuni=1, nlay=6)  # lenuni=1 for feet

# Assign the structured grid to the model and update the model grid
ml = get_model(ModelPath)  # load model using function above
ml.modelgrid = mg # get model grid
ml.update_modelgrid() 

#==================================LOAD POLYLINE INPUT========================


ff=gpd.read_file(StreamsFile) # load shapefile 
print("ORIGINAL CRS: ")
print(ff.crs) # print crs. VERY IMPORTANT THAT IT MATCHES EXPECTATION AND IS IN EXPECTED UNITS! 

ff[Routing_Field] = ff[Routing_Field].replace('-9999', '0') # in GIS, exit segments are sometimes designated -999. We want this to be 0 instead (leaving the model)
#ff = ff.to_crs(2226) # If you want to covert coordinate system. but, be warned that it probably wont change the unit association that SFRmaker reads internally. 
#print("PROJECTED TO : ")
#print(ff.crs)

# Check if all toid values have corresponding ids
missing_connections = set(ff[Routing_Field]) - set(ff[ID_Field]) # check for missing routing. fairly common issue. 
if missing_connections:
    print(f"Missing connections for toid: {missing_connections}")
    print("ANY VALUE OTHER THAN 0 IS A CRITICAL ERROR. FIX THIS")
else:
    print("No missing connections, thats good!")
ff[Routing_Field] = pd.to_numeric(ff[Routing_Field], errors='coerce').fillna(0).astype(int)
ff[ID_Field] = pd.to_numeric(ff[ID_Field], errors='coerce').fillna(0).astype(int)
print("Streams file loaded successfully. ")
#print(ff.head())

print("Converting units")
#length conversions
ff[LengthField]=ff[LengthField]*LengthConversion
ff[ArbolateField2]=ff[ArbolateField2]*LengthConversion
#height conversions
ff[UpperElevationField]=ff[UpperElevationField]*ElevationConversion
ff[LowerElevationField]=ff[LowerElevationField]*ElevationConversion
crs_2226 = CRS.from_epsg(2226)
crs_2226 = pyproj.CRS.from_epsg(2226)  # Create a CRS object for EPSG:2226

print("Calculating width from arbolate sum")
ff['Width_2']=Custom_width_from_arbolate_sum(ff[ArbolateField2], a=0.1193, b=0.5032, minimum_width=3., input_units='feet',output_units='feet') # we feed in the arbolate sum at the bottom of the segment
ff['Width_1']=Custom_width_from_arbolate_sum(ff[ArbolateField2]-ff[LengthField], a=0.1193, b=0.5032, minimum_width=3., input_units='feet',output_units='feet') # we feed in the arbolate sum at the bottom of the segment, minus the legnth (equivalent to arbolate sum of the top)

lines = sfrmaker.Lines.Custom_from_dataframe(ff, id_column=ID_Field,
                                  routing_column=Routing_Field,
                                  width1_column='Width_1',  # calculated above
                                  width2_column='Width_2',
                                  arbolate_sum_column2=ArbolateField2, # 
                                  up_elevation_column=UpperElevationField,
                                  dn_elevation_column=LowerElevationField,
                                  name_column='GNIS_Name',
                                  geometry_column='geometry',
                                  asum_units=LengthUnits, # length units for 'length' if provided and Arbolate sum if not
                                  elevation_units=HeightUnits,  # There should be an internal conversion in SFRMaker. end product should be 'feet'. 
                                  hydraulic_conductivity_column=ConductivityField,  # Specify conductivity column here. Should be in model units already
                                  crs=crs_2226, model_length_units='feet', width_units='feet') # width units important to assign. see lines.py for the inputs

print(lines.geometry_length_units) # This is important -- make sure this is assigned properly...
print(lines._geometry_length_units) # This is important -- make sure this is assigned properly...

print("Lines created successfully from dataframe.")
#print(lines.df[['id', 'strhc1']].head())  # just making sure strhc1 didn't get overwritten by SFR
#==================================CREATE SFR IN PYTHON ========================
print("CREATING SFR DATA")
sfrdata = lines.to_sfr(model=ml, model_length_units='feet',crs=crs_2226)

#==================================Merge Conductivity Values========================
print('MERGING CONDUCTIVITY VALUES')
# SFRmaker REALLY wants to have strch1 =1. Despite everything I try, it loves to assign this value internally.
#rather than create more custom functions, I just manually join strch1 back into the sfrdata here. 
# Merge the 'strhc1' field from lines.df into sfrdata.reach_data based on 'line_id'
merged_df = pd.merge(sfrdata.reach_data, lines.df[['id', 'strhc1']], left_on='line_id', right_on='id', how='left')

# After merging, we have 'strhc1_x' (from reach_data (all 1)) and 'strhc1_y' (from lines.df correct)
# Rename 'strhc1_y' to 'strhc1' and drop 'strhc1_x' if necessary
if 'strhc1_y' in merged_df.columns:
    merged_df['strhc1'] = merged_df['strhc1_y']  # Rename 'strhc1_y' to 'strhc1'
    merged_df.drop(columns=['strhc1_x', 'strhc1_y'], inplace=True)  # Drop unnecessary columns

# Fill NaN values in 'strhc1' with a default value of 1.0. Don't think there will be any though. 
merged_df['strhc1'] = merged_df['strhc1'].fillna(1.0)

# Assign the modified dataframe back to sfrdata.reach_data
sfrdata.reach_data = merged_df

# Check if 'strhc1' was properly assigned
print(sfrdata.reach_data[['line_id', 'strhc1','rchlen']].head())  # Final check
sfrdata_df=sfrdata.reach_data # for viewing
#==================================WRITE SHAPEFILES AND PACKAGE========================
print('WRITING SHAPEFILES')
sfrdata.write_shapefiles()
cells = gpd.read_file(r"Q:\Work_Files\9400_Sonoma\2024UpdateExpandGridOWHM2\UpdateSFR\shps\modflowtest_sfr_cells.shp")
lgdf = gpd.read_file(r"Q:\Work_Files\9400_Sonoma\2024UpdateExpandGridOWHM2\UpdateSFR\shps\modflowtest_sfr_lines.shp")
outlet = gpd.read_file(r"Q:\Work_Files\9400_Sonoma\2024UpdateExpandGridOWHM2\UpdateSFR\shps\modflowtest_sfr_outlets.shp")

m = cells.explore()
lgdf.explore(m = m, color = 'k')
outlet.explore(m=m, style_kwds = {'color':'green'})
print('WRITING PACKAGE')
sfrdata.write_package()




