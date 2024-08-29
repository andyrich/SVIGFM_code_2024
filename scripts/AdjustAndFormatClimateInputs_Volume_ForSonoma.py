# -*- coding: utf-8 -*-
"""
@author: pwickham
Script Name: AdjustAndFormatClimateInputs_Volume.py
Author: Patrick Wickham
Email: pwickham@elmontgomery.com
Created: July 2024
Updated: August 2024
    Developed by 2024 Montgomery and Associates Water Resource Consultants
    This work is distributed use in for internal M&A and partner agency 
    modeling efforts, without warranty of any kind. 
    
    For questions about utilizing this script contact Patrick Wickham
    pwickham@elmontgomery.com

OVERVIEW
This script reads a database format series of node, row, col, and value data. 
It then performs adjustments related to unit conversion and in/out of model area
then pivots for MODFLOW input. 
It performs this for each month in the data

THEORY
Script pivots the row, col, and zone fields to turn database format into matrix
also converts units and ajusts for subcatchment in/out of model area

MECHANICS
    Simple pandas and numpy

DEVELOPMENT NOTES
    model cell is 500x500ft 
    cleaned comments and old code to provide to client

"""
#===================================USER INPUTS==============================
DatabaseFiles=[r'G:\GIS-Tuc\Projects\9400_SonomaValley\Analysis\BCM_Update_2024\SVIGFM_PET_1969to2023.csv', #precip. Files with row/col/ and value. value = 'Z' 
               r'G:\GIS-Tuc\Projects\9400_SonomaValley\Analysis\BCM_Update_2024\SVIGFM_PPT_1969to2023.csv.csv' # pet
               ]
names=['ppt','pet'] # names for the type of data, if conducting in loop. Here we loop over precip and pet

conversions=[1/304.8, #mm to feet. needed for all database types (identical here)
             1/304.8] 

AdjustmentFiles=[r'O:\Work_Files\9400_Sonoma\2024UpdateExpandGridOWHM2\UpdatePrecipEt_BCMV8\Subbasins_PET_1969to2023_format.csv', # files with total subcatchment value, used for adjusting the in-model subcatchment fluxes to reflect the total subcatchment volume
                r'O:\Work_Files\9400_Sonoma\2024UpdateExpandGridOWHM2\UpdatePrecipEt_BCMV8\Subbasins_PET_1969to2023_format.csv' ]

SubcatchmentIdentification=r'O:\Work_Files\9400_Sonoma\2024UpdateExpandGridOWHM2\UpdatePrecipEt_BCMV8\SVIGFM_DIS_Subcatchments_GRIDCODE.csv' # file with node, row, col, Gridcode. Gridcode is set to 9999 if not a subcatchment 

# days per month
fn_days = r'O:\Work_Files\9400_Sonoma\2024UpdateExpandGridOWHM2\UpdatePrecipEt_BCMV8\days_per_month_PET.csv'

# where to export array files
ExportFolders=[r'O:\Work_Files\9400_Sonoma\2024UpdateExpandGridOWHM2\UpdatePrecipEt_BCMV8\ExportFolder\v2\pet\\',
               r'O:\Work_Files\9400_Sonoma\2024UpdateExpandGridOWHM2\UpdatePrecipEt_BCMV8\ExportFolder\v2\pet\\'
]               

ZoneFile=r'O:\Work_Files\9400_Sonoma\2024UpdateExpandGridOWHM2\UpdatePrecipEt_BCMV8\GHBCells.csv' # array, for excluding certain zones. here we exclude zone 1 (zonebudget zone 13).
ZonesToZeroOut=1 # GHB. No flux on GHB cells

SubcatchmentsOfInterest=[4888,237,5836,6316,1106,7114,100,320,4869] # specify the subbasins that are cut off by the model area. You can also have it run over all the subcatchments, but faster to just specify (adjustment will be multiplying by 1, aka no effect.)

#============================================================================

#Required Modules
import pandas as pd    
import numpy as np

#=====================Initiate Loop and Read Data ========================== 
SubcatchmentIdentification=pd.read_csv(SubcatchmentIdentification)
GHBZones=pd.read_csv(ZoneFile)
days_per_month = pd.read_csv(fn_days)
i=0 # changed to read in PET
for file in DatabaseFiles:  # load all the applicable data
    raw_value_data = pd.read_csv(file)
    value_name = names[i]
    AdjustmentBySubcatchment = pd.read_csv(AdjustmentFiles[i])  # file with sum of subcatchment value for each date. used for adjusting in-model values to account for out of model flows
    print("Processing " + value_name + " Data")
    all_value_data = pd.merge(raw_value_data, SubcatchmentIdentification[['node', 'GRIDCODE']], on='node')  # merge to get GRIDCODE for subcatchment
    all_value_data = pd.merge(all_value_data, GHBZones[['node', 'GHB']], on='node')  # merge to get GHB binary

    #===================QC GIS Calculations and Adjust Values===================
    # initialize arrays. These are mostly for QC besides 'dates'
    GISVolumeLog=[]
    CellularAggregationVolumeLog=[]
    GISAreaLog=[]
    CellularAggregationAreaLog=[]
    ConversionFactor=conversions[i]
    GISMeanFluxLog=[]
    CellularAggregationMeanFluxLog=[]
    dates=all_value_data['Source'].unique()
     
    for date in all_value_data['Source'].unique():  # for each unique date. 'Source'=GIS date column (month)
        print(date)
        value_data = all_value_data[all_value_data['Source'] == date].copy() # copy dataframe
        days = days_per_month.loc[days_per_month['Source'] == date, 'days'].iloc[0] # extract number of days
    
        for Subcatchment in value_data['GRIDCODE'].unique():  # Made adjustment for subcatchment areas
            #if isinstance(Subcatchment, (int, float)) and not np.isnan(Subcatchment): #this loop can be used to test the workflow. If printed disrepencies are quite large for catchments fully inside the model, then there is probably a unit conversion problem
            if Subcatchment in SubcatchmentsOfInterest:  # if the area is a subcatchment we need to adjust, e.g cut off by the model grid
                print('ADJUSTING '+str(Subcatchment))    
                
                InModelArea=500*500*len(value_data[value_data['GRIDCODE'] == Subcatchment]) # calculate approximate area of simulated subcatchment in acres, by summing up cells.
                FullArea=AdjustmentBySubcatchment[(AdjustmentBySubcatchment['Source'] == date) & (AdjustmentBySubcatchment['GRIDCODE'] == Subcatchment)]['AREA'].iloc[0]# # pull area in acres from subcatchment subsummary file. 
                GISAreaLog.append(FullArea) 
                CellularAggregationAreaLog.append(InModelArea)
                
                InModelMeanFlux_length=value_data[value_data['GRIDCODE'] == Subcatchment]['Z'].mean() # calculate average flux (length) in the simulated area 
                FullAreaMeanFlux_length=AdjustmentBySubcatchment[(AdjustmentBySubcatchment['Source'] == date) & (AdjustmentBySubcatchment['GRIDCODE'] == Subcatchment)]['MEAN'].iloc[0] # pull mean flux
                GISMeanFluxLog.append(FullAreaMeanFlux_length) 
                CellularAggregationMeanFluxLog.append(InModelMeanFlux_length) 
                
                InModelVolume=InModelArea*InModelMeanFlux_length # this volume is in mm/cubic feet 
                FullSubbasinVolume=FullArea*FullAreaMeanFlux_length
                
                if FullAreaMeanFlux_length != 0 and not np.isnan(FullAreaMeanFlux_length): # calculate ratio of volume flux inside/outside model
                    PercentVolumeOutsideModel = InModelVolume / FullSubbasinVolume if FullSubbasinVolume != 0 else 1
                else:
                    PercentVolumeOutsideModel = 1  # handle the case where SubcatchmentSumValue is 0 or NaN
                
                # calcuate volume flux. Determine volume fluc ratio. Adjust according to volume flux ratio. 
                test=AdjustmentBySubcatchment[(AdjustmentBySubcatchment['Source'] == date) & (AdjustmentBySubcatchment['GRIDCODE'] == Subcatchment)] # record for viewing and debugging in interpreter
                test2=value_data[value_data['GRIDCODE'] == Subcatchment]
                
                GISVolumeLog.append(FullSubbasinVolume)
                CellularAggregationVolumeLog.append(InModelVolume)
                if not np.isnan(PercentVolumeOutsideModel) and PercentVolumeOutsideModel != 0:
                    value_data.loc[value_data['GRIDCODE'] == Subcatchment, 'Z'] = (value_data.loc[value_data['GRIDCODE'] == Subcatchment, 'Z'] / PercentVolumeOutsideModel )  # adjust precip

                else:
                    print(f"Skipping adjustment for Subcatchment {Subcatchment} due to invalid PercentFluxOutsideModel: {PercentVolumeOutsideModel }")
                    print("POTENTIAL ERROR, OR DUE TO 0 FLUX IN SUBCATCHMENT")
                    continue
                NewInModelFluxLength=value_data[value_data['GRIDCODE'] == Subcatchment]['Z'].mean() # calculate average flux (length) in the simulated area 
                NewInModelVolume=InModelArea*NewInModelFluxLength # calculate the actual precip sum for the subcatchment after adjustment
                  
                print("value increased by factor of " + str(NewInModelVolume / InModelVolume if InModelVolume != 0 else float('inf')))
                print("Finished adjusting Subcatchment " + str(Subcatchment))
                
                if (FullSubbasinVolume - NewInModelVolume > 0.1) or (FullSubbasinVolume - NewInModelVolume < -0.1):  # check significant discrepancy
                    print("WARNING!!! Discrepancy is " + str(FullSubbasinVolume - NewInModelVolume)) # shouldn't happen unless something is wrong
                
        #Convert to feet /day (from mm/month)
        value_data.loc[:, 'Z'] = value_data['Z'] * ConversionFactor / days # convert from mm/month to ft/d
                    
        #exclude zones where zero is needed (GHB)
        value_data.loc[value_data['GHB'] == ZonesToZeroOut, 'Z'] = (0)
        
        # database format to array
        matrix = value_data.pivot(index='row', columns='column_', values='Z')  # Pivot the DataFrame to transform into a matrix
        total_na_matrix = matrix.isna().sum().sum()  # check NA
        
        if total_na_matrix > 0:
            matrix = np.nan_to_num(matrix, nan=0)  # replace nan with 0 if it exists. 
        
        # export
        ExportFolder = ExportFolders[i]
        filepath = ExportFolder +value_name+ date + '.txt'
        np.savetxt(filepath, matrix, delimiter=" ", fmt='%f') # may need to change delimeter based on modflow formatting.
    
    i += 1  # iterate over calculation type