# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 22:54:02 2024

    @author: pwickham
    Script Name: BuildHOB.py
    Author: Patrick Wickham
    Email: pwickham@elmontgomery.com
    Created: September 2024
    Updated: September 2024
        Developed by 2024 Montgomery and Associates Water Resource Consultants
        This work is distributed without warranty of any kind. 
        
        For questions about utilizing this script contact Patrick Wickham
        pwickham@elmontgomery.com
    
    OVERVIEW
    This script creates the hob file. It takes 4 input files:
            Well List file: Contains station name, layering, Row, Col, and XY. layering percentages not calculated here. Calculated in other scripts like 'calculatethickness.py' 
            Centroids file: centroid XY exported from in GIS. Used for calcuting ROFF and COFF
            Observations File: file with all observations. Can be screened and/or averaged to stress period, using the user inputs below
            StressPeriodFile: File with stress periods and corresponding dates. 
            All inputs are assumed to be from GIS, and therfore pre-rotated. script accounts for un-rotating these so it can properly calulcate ROFF and COFF.
    THEORY
        https://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/hob.html
        
    MECHANICS
        pandas

    DEVELOPMENT NOTES
        script to build HOB. 
        Added functionality to screen for dates (on top of existing functionality that screens for TOFF  >31 )
        Added optional Y/N functionality to average observations by stress period. 
"""
import os
import flopy
print(os.getcwd())
#===================================USER INPUTS==============================
export_file=r'HOB20250910.hob' #where to print HOB
WellListFile=r'WellDetails_20240911.xlsx' #File with WellName, Row, Col, X,Y, and percentage layering
CentroidsFile=r'Centroids.csv' #file with row , col, rowcol, centroidX, CentroidY
ObservationsFile=r'WaterLevelsForHOB.csv' # file with observations 'value' and date.
AverageObservationsBySP='Y' # Y/N for whether to average the observations by SP (month). 
StressPeriodFile=r'SP_Date_Lookup_SVIGFM_Hist.xlsx' # file with stress periods and date
Start= '1969-12-01' # first date that you want to include in the HOB (start date) format = YYYY-MM-DD
End= '2018-09-30'  # last date that you want to include in the HOB (end date)format = YYYY-MM-DD
HOBDRY=-9999 # dry well value 
IUHOBSV=88 # unit number
MAXM=6
TomulthLine='1       TOMULTH EVH '
ITTLine='    1        ITT'
layer_columns = ['Layer_1', 'Layer_2', 'Layer_3', 'Layer_4', 'Layer_5', 'Layer_6'] # layer columns in well excel file. must have numberic 1-n somewhere in the name

#Required Modules
import pandas as pd    
import winsound
import numpy as np

# Grid parameters
delr, delc = 500, 500
xul = 6382956.489134505 # x origin coord
yul = 1918132.341874674  # y origin coord
angrot = 23.0  # Rotation angle in degrees
theta = np.radians(angrot)  # Convert rotation angle to radians
StartDate=pd.to_datetime(Start)
EndDate=pd.to_datetime(End)
#==================================DEFINE FUNCTIONS========================

def FinishedBeep(): # a function to inform user when script has finished running or completed critical steps (-Patrick Wickham)
    winsound.Beep(293, 200)
    winsound.Beep(293, 200)
    winsound.Beep(450, 600)
    winsound.Beep(350, 600)
    winsound.Beep(450, 600)
    winsound.Beep(600, 600)

#==================================LOAD WELL LAYERING DATABASE========================
print("LOADING DATA")
Wells=pd.read_excel(WellListFile)

#==================================LOAD OBSERVATION DATABASE========================
Observations=pd.read_csv(ObservationsFile)

#===============================LOAD CENTROIDS===========================
Centroids=pd.read_csv(CentroidsFile)

#==================================LOAD STRESS PERIODS========================
StressPeriods=pd.read_excel(StressPeriodFile)

#==================================CALCULATE REQUIRED INPUTS========================
print("CALCULATING REQUIRED INFORMATION")
#Join Observations To stress periods. calculate TOFFSET
Observations['Timestamp'] = pd.to_datetime(Observations['Timestamp'])
StressPeriods['month_start'] = pd.to_datetime(StressPeriods['Date'])
StressPeriods['month_end'] = StressPeriods['month_start'] + pd.offsets.MonthEnd(1)
Observations = pd.merge_asof(Observations.sort_values('Timestamp'), StressPeriods.sort_values('month_start'), left_on='Timestamp',  right_on='month_start',  direction='backward')
Observations['TOFFSET'] = (Observations['Timestamp'] - Observations['month_start']).dt.days # calculate time difference in days. used for initial screeening. 
Observations = Observations[Observations['TOFFSET'].abs() <= 31] # Filter out observations where the time difference is greater than 31 days (1 month). useful for ditching everyinth not in a stress period
Observations = Observations[(Observations['Timestamp'] >= StartDate) & (Observations['Timestamp'] <= EndDate)]

# average to SP if requested 
if AverageObservationsBySP.lower() == 'y' : # if requested, get the average for multiple dates that occur in each stress period
    Observations= Observations.groupby(['sp', 'station_name']).agg({
    'Timestamp': 'mean',     'Value': 'mean',    'Date': 'mean',    # specify columns to mean and columns to retain. 
    'month_start': 'first','month_end': 'first',       # Retain first non-numeric value
}).reset_index()
    
Observations['TOFFSET'] = (Observations['Timestamp'] - Observations['month_start']).dt.days # recalculate TOFFset following aggregation, in case aggregation took place.

#row and col offset
Wells['Node'] = Wells['Node'].astype(int)
Centroids['SVIGFM_DIS_Project_node'] = Centroids['SVIGFM_DIS_Project_node'].astype(int)
Wells =  pd.merge(Wells.sort_values('Node'), Centroids, left_on='Node',  right_on='SVIGFM_DIS_Project_node')
Wells['well_x_unrot'] = (Wells['Easting_x'] - xul) * np.cos(theta) + (Wells['Northing_y'] - yul) * np.sin(theta) + xul # unrotate 
Wells['well_y_unrot'] = -(Wells['Easting_x'] - xul) * np.sin(theta) + (Wells['Northing_y'] - yul) * np.cos(theta) + yul
Wells['centroid_x_unrot'] = (Wells['POINT_X'] - xul) * np.cos(theta) + (Wells['POINT_Y'] - yul) * np.sin(theta) + xul
Wells['centroid_y_unrot'] = -(Wells['POINT_X'] - xul) * np.sin(theta) + (Wells['POINT_Y'] - yul) * np.cos(theta) + yul

# Calculate ROFF and COFF
Wells['ROFF'] = round((Wells['well_y_unrot'] - Wells['centroid_y_unrot']) / delc,3)
Wells['COFF'] = round((Wells['well_x_unrot'] - Wells['centroid_x_unrot']) / delr,3)

HOB_Database=pd.merge(Observations.sort_values('station_name'), Wells, on='station_name') # database with everything

HOB_Database['NumAssignedLayers'] = (HOB_Database[layer_columns] > 0).sum(axis=1) # count number of layers with a non-0 layering assigned
NH = len(HOB_Database)# NH is the total number of observations
MOBS = (HOB_Database['NumAssignedLayers'] > 1).sum()# MOBS is the number of observations that have more than one layer assigned

HOB_Database = HOB_Database.sort_values(by=['station_name', 'Timestamp'])

# Create a unique ID based on the station name and a running count of observations
HOB_Database['unique_id'] = HOB_Database.groupby('station_name').cumcount() + 1
HOB_Database['unique_id'] = HOB_Database['station_name'] + '_' + HOB_Database['unique_id'].astype(str)
import os
print(os.getcwd())
#==================================ASSEMBLE HOB========================

# Assuming you have a DataFrame df with columns Layer_1 to Layer_6
columns = ['Layer_1', 'Layer_2', 'Layer_3', 'Layer_4', 'Layer_5', 'Layer_6']

# Step 1: Sum the values across the specified columns by row
row_sums = HOB_Database[columns].sum(axis=1)

# Step 2: Divide each row by its corresponding row sum
HOB_Database[columns] = HOB_Database[columns].div(row_sums, axis=0)

# print dataset 1 and 2 (header)
with open(export_file, 'w') as file:
    print("PRINTING HEADER")
    header_line = f"{NH} {MOBS} {MAXM} {IUHOBSV} {HOBDRY}  NH MOBS MAXM IUHOBSV HOBDRY\n"
    file.write(header_line)
    file.write(TomulthLine + '\n')

#print dataset 3
    print("PRINTING LAYERING AND OBSERVATION INFO")
    for well in HOB_Database['station_name'].unique():
        print("               "+ str(well))
        df=HOB_Database[HOB_Database['station_name']==well]
        NumLayers=df.iloc[0]['NumAssignedLayers']
        ExampleLine=df.iloc[0] #pull one row to make things easier
        row=ExampleLine['Row']
        col=ExampleLine['Col']
        sp=ExampleLine['sp']
        ROFF=ExampleLine['ROFF']
        COFF=ExampleLine['COFF']
        NumObs=-len(df)
        NegativeNumLayers=NumLayers*-1
        FirstMeasurmentInHead=ExampleLine['Value'] # pull first value
        FirstTOFF=ExampleLine['TOFFSET'] # pull first value. 
        if NumLayers>1:
            print("               "+"    (multi layer well)")
            file.write(f"{well} {NegativeNumLayers} {row} {col} {NumObs} {FirstTOFF} {ROFF} {COFF} {FirstMeasurmentInHead} \n")
            layers = [] # Identify layers with non-zero percentages
            for i in range(1, MAXM+1):  # Assuming 6 layers
                if ExampleLine[f'Layer_{i}'] > 0:
                    layers.append((i, ExampleLine[f'Layer_{i}']))
            multilayer_string = " ".join(f"{layer} {percentage:.16f}" for layer, percentage in layers)
            file.write('   '+multilayer_string + '    MLAY(1), PR(1), MLAY(2), PR(2), ..., MLAY(|LAYER|), PR(|LAYER|)'+ '\n')
        else:
            print("               "+"     (single layer well)")
            layer_column = [col for col in ['Layer_1', 'Layer_2', 'Layer_3', 'Layer_4', 'Layer_5', 'Layer_6'] if ExampleLine[col] == 1]           
            if layer_column:
                layer = int(layer_column[0].split('_')[1])  # Extract the layer number from the column name (e.g., 'Layer_5' -> 5)
            else:
                print("ERROR! LAYER NOT DEFINED FOR "+ str(well))
                print("    this shouldn't happen. check your layering inputs")
            file.write(f"{well} {layer} {row} {col} {NumObs} {FirstTOFF} {ROFF} {COFF} {FirstMeasurmentInHead} \n")
        file.write(ITTLine + '\n')
        
        # print observations for this well
        for _, entry in df.iterrows():
            ID=entry.unique_id
            sp=entry.sp
            TOFF=entry.TOFFSET
            head=entry.Value
            date=entry.Timestamp.strftime('%m/%d/%Y')
            file.write(f"     {ID} {sp} {TOFF} {head} {date} \n")            
            
# Can QC using flopy if you'd like. as an example:
m = flopy.modflow.Modflow()
hobs = flopy.modflow.ModflowHob.load(export_file, m)
            
FinishedBeep()












