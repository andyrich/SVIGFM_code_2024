# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:25:13 2024

    @author: pwickham
    Script Name: BuildSFR.py
    Author: Patrick Wickham
    Email: pwickham@elmontgomery.com
    Created: October 2024
    Updated: October 2024
        Developed by 2024 Associates Water Resource Consultants
        This work is distributed without warranty of any kind. 
        
        For questions about utilizing this script contact Patrick Wickham
        pwickham@elmontgomery.com
    
    OVERVIEW
    Simple script to read and plot sfr attributes found in SV_sfr_sim.dat
    you must set the values 	ISTCB1 and ISTCB2 to have these printed.
    Its useful for looking at by-reach and by-segment flow information
    
    THEORY
        https://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/sfr.html
        
    MECHANICS
        pandas 
        
    DEVELOPMENT NOTES
        `Currently set up to extract and process information, and plot htmls and jpgs.
        `Added earlier observation data for USGS gauges like kenwood, agua caliente, and Nathanson
        `Question for Andy -- how are 0 flows handled in the log normalization? Constant added? Right now plots will just drop them and throw a soft error
        

"""
export_directory=r'Q:\Work_Files\9400_Sonoma\2024UpdateExpandGridOWHM2\UpdateSFR\PostProcessSept2024_HOBsSFR_testLengthFix\SFR_Flows\\'  #Where to post output. include 2 slashes '\\' at end 
File2Process=r'Q:\Work_Files\9400_Sonoma\SVIGFM_Backups\Sept2024_HOBsSFR_testLengthFix\output\SV_sfr_sim.dat' # output filw
ObservationData=r'Q:\Work_Files\9400_Sonoma\2024UpdateExpandGridOWHM2\UpdateSFR\SFRObservations\CombinedDischargeObservationData.xlsx' # observation data in database format
Times=r'\\oak-model\public\Work_Files\9400_Sonoma\2024UpdateExpandGridOWHM2\DevelopZonebudget\SP_Date_Lookup_SVIGFM_Hist.xlsx' # SP, Date, and DaysInMonth
htmlPlots='y' # Y/N binary for plotting html bar plots with all reach-level flows and obs
CalibPlots='y' # Y/N binary for plotting lines of flow_in and observations
GaugeLocations={ # dictionary of gauge locations in format 'Name': ['row_col']
'Sonoma Creek at Agua Caliente Rd': ['100_38'],
'Sonoma Creek at Verano Ave': ['116_39'],
'Sonoma Creek at Leveroni Rd': ['136_37'],
'Sonoma Creek at Watmaugh Rd': ['144_36'],
'Sonoma Creek at Hwy 12': ['15_34'],
'Sonoma Creek at Kenwood Gauge': ['20_30'],
'Sonoma Creek At Randolph Rd': ['25_33'],
'Sonoma Creek at Warm Springs/Lawndale Rds': ['34_30'],
'Sonoma Creek Above Yulupa Creek confl': ['49_24'],
'Sonoma Creek at Warm Springs Rd': ['59_29'],
'Sonoma Creek Above Calabasas Creek confl': ['66_34'],
'Sonoma Creek at Madrone Rd': ['86_36'],
'Unnamed trib to Sonoma Creek Above Kenwood Gauge': ['19_29'],
'Yulupa Creek at Warm Springs Rd': ['49_24'],
'Asbury Creek at Arnold Dr': ['72_32'],
'Stuart Creek At Arnold Dr': ['63_38'],
'Agua Caliente Creek at Sonoma Creek': ['116_39'],
'Carriger Creek at Arnold Street Bridge': ['125_29'],
'Trib at Warm Springs Rd': ['32_32'],
'Calabasas Creek at Hwy 12': ['51_44'],
'Calabasas Creek at Dunbar Rd': ['53_42'],
'Calabasas Creek on Warm Springs Rd': ['65_34'],
'Calabasas Creek above Sonoma Creek confl': ['66_34'],
'Arroyo Seco Creek at Denmark St': ['146_61'],
'Arroyo Seco at Hyde Burndale Rd': ['153_56'],
'Arroyo Seco Creek at E. Napa St': ['135_60'],
'Carriger Creek at Leveroni Rd': ['135_34'],
'Dowdall Creek at Riverside Dr': ['118_37'],
'Felder Creek at private residence': ['133_22'],
'Felder Creek at cattle guard': ['133_18'],
'Felder Ck at Leveroni Rd': ['135_30'],
'Graham Creek At Warm Springs Rd': ['59_28'],
'Hooker Creek at Hwy 12': ['90_43'],
'NathansonCreek at Broadway': ['156_42'],
'Nathanson Creek at Nature Park': ['138_45'],
'Nathanson Creek at Patten St': ['131_49'],
'Rodgers Creek at Watmaugh Rd': ['144_23'],
'Rodgers Creek at via Columbard': ['137_18']
                }

OtherStatsAndPlots='n' # I had some code I found useful for looking at outputs. Feel free to delete if you'd like. 

#ConversionFactor=(1/43560) #Multiplication factor to convert budget output (ex for cubic feet to acre-feet input (1/43560) 

import pandas as pd    
import re
#import os
if htmlPlots.lower()=='y' or CalibPlots.lower()=='y':
    import plotly.graph_objects as go
import winsound
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
#import numpy as np
#import time
#import sys 


#==================================DEFINE FUNCTIONS=========================
def FinishedBeep(): # a function to inform user when script has finished running or completed critical steps (-Patrick Wickham)
    winsound.Beep(293, 200)
    winsound.Beep(293, 200)
    winsound.Beep(450, 600)
    winsound.Beep(350, 600)
    winsound.Beep(450, 600)
    winsound.Beep(600, 600)

def ProcessSFROutputFile(filepath): # process the sfr output file to read its clunky format
    data = []
    current_period = None
    print("READING: "+filepath)
    # Open the file for reading
    with open(filepath, 'r') as file:
        for line in file:
            # Look for the period header
            period_match = re.search(r'PERIOD\s+(\d+)', line)
            if period_match:
                current_period = int(period_match.group(1))
                #print("NEW PERIOD: " + str(current_period))
            
            # Check if the line is data (contains numbers and follows the anticipated format)
            data_match = re.match(r'\s*\d+', line)
            if data_match and current_period is not None:
                # Split the line into fields (space-separated values)
                fields = re.split(r'\s+', line.strip())
                # Add the current period to the data
                fields.append(current_period)
                # Append the row of data to the list
                data.append(fields)
                
    # Adjust the column names to match the data s # you may need to change this based on what prints from your file
    column_names = ['LAYER', 'ROW', 'COL', 'SEG', 'RCH', 'FLOW_INTO_RCH', 'FLOW_TO_AQUIFER', 'FLOW_OUT_OF_RCH', 
                    'OVRLND_FLOW', 'DIRECT_PRECIP', 'STREAM_ET', 'STREAM_HEAD', 'STREAM_DEPTH', 'STREAM_WIDTH', 
                    'STREAMBED_COND', 'STREAMBED_GRADIENT', 'STREAMBED_ELEV', 'PERIOD']
    df = pd.DataFrame(data, columns=column_names)
    return df    
#==================================START PROCESSING DATA========================= 
# file to dataframe
df_raw = ProcessSFROutputFile(File2Process)
df=df_raw.copy(deep=True)
df = df.apply(pd.to_numeric, errors='coerce')

len1=len(df_raw)
len2=len(df)
missingdata=len1-len2
if abs(missingdata)>0:
    print("WARNING! DATA MAY HAVE BEEN LOST IN CONVERSION TO NUMERIC VALUES") # shouldn't happen but worth checking 

FullModelTimeseries=  df.groupby(['PERIOD'],as_index=False).sum(numeric_only=True) # group the whole model (total flow)
SegmentTimeseries=  df.groupby(['SEG','PERIOD'],as_index=False).sum(numeric_only=True) # group by segment (total flow)
SegmentTimeseriesMean=df.groupby(['SEG','PERIOD'],as_index=False).mean(numeric_only=True)  # group by segment (mean flow)
ReachTimeseries=  df.groupby(['SEG','RCH','PERIOD'],as_index=False).mean(numeric_only=True) # group by reach (mean flow) -- not that this row shouldn't practically introduce any change from df to ReachTimeseries. But it does make a copy. 
ReachTimeseries['SEG_RCH']=ReachTimeseries['SEG'].astype(int).astype(str)+"_"+ReachTimeseries['RCH'].astype(int).astype(str) # unique ID for reach
ReachTimeseries['ROW_COL']=ReachTimeseries['ROW'].astype(int).astype(str)+"_"+ReachTimeseries['COL'].astype(int).astype(str) # unique ID for reach by row/col

#join to SP dataframe and calculate flow in CFS (from CFD)
Dates_DF=pd.read_excel(Times)
ReachTimeseries=pd.merge(ReachTimeseries,Dates_DF,left_on='PERIOD',right_on='sp')
Columns2Convert=['FLOW_INTO_RCH', 'FLOW_TO_AQUIFER', 'FLOW_OUT_OF_RCH', 
                'OVRLND_FLOW', 'DIRECT_PRECIP', 'STREAM_ET']
for column in Columns2Convert:    ReachTimeseries[column] = ReachTimeseries[column] / (60*60*24)

Observation_df=pd.read_excel(ObservationData)
    
#==============================GATHER AND PLOT AT GAUGE LOCATIONS=========================
print("PRINTING BY REACH")
CombinedSimulatedGaugeData=pd.DataFrame()
for gauge_name, row_col in GaugeLocations.items():
    UniqueID = row_col[0]  # Extract the 'row_col' from the dictionary
    if UniqueID in ReachTimeseries['ROW_COL'].unique():
        clean_gauge_name = gauge_name.replace(" ", "_").replace("/", "_")
        print("PRINTING "+str(gauge_name))
        segment_rchdf=ReachTimeseries[ReachTimeseries['ROW_COL']==UniqueID].copy()
        obs=Observation_df[Observation_df['SiteName']==gauge_name]
        
        if htmlPlots.lower()=='y':
            fig = go.Figure()
            fig.add_trace(go.Bar( x=segment_rchdf['Date'], y=-segment_rchdf['FLOW_TO_AQUIFER'],name=str("FLOW_TO_AQUIFER")))   
            fig.add_trace(go.Bar( x=segment_rchdf['Date'], y=-segment_rchdf['FLOW_OUT_OF_RCH'],name=str("FLOW_OUT_OF_RCH"))) 
            fig.add_trace(go.Bar( x=segment_rchdf['Date'], y=segment_rchdf['FLOW_INTO_RCH'],name=str("FLOW_INTO_RCH"))) 
            fig.add_trace(go.Bar( x=segment_rchdf['Date'], y=segment_rchdf['OVRLND_FLOW'],name=str("OVRLND_FLOW"))) 
            fig.add_trace(go.Bar( x=segment_rchdf['Date'], y=segment_rchdf['DIRECT_PRECIP'],name=str("DIRECT_PRECIP")))  
            fig.add_trace(go.Bar( x=segment_rchdf['Date'], y=segment_rchdf['STREAM_ET'],name=str("STREAM_ET")))  
        
            if len(obs)>0:
                #print("    HAS OBSERVATIONS")
                fig.add_trace(go.Scatter(x=obs['Month'], y=obs['Discharge'], name="Discharge Observations", mode='markers')) 
        
            fig.update_layout(title=gauge_name,xaxis_title='Stress Period',yaxis_title='Flows, Model Units (Ft^3/s)',legend_title='flow',barmode='relative')
            fig.write_html(export_directory+str(clean_gauge_name)+'.html')
            #fig.write_image(export_directory+str(clean_gauge_name)+'.jpg', scale=5)
            #fig.write_image(export_directory+str(clean_gauge_name)+'.jpg', scale=5)  # requires kaleido dependency (e.g. conda install kaleido)
        if CalibPlots.lower()=='y':
            fig = go.Figure()   
            fig.add_trace(go.Scatter( x=segment_rchdf['Date'], y=np.log(segment_rchdf['FLOW_INTO_RCH']),name=str("FLOW_INTO_RCH"),line=dict(color='blue', width=2, dash='solid')))   
        
            if len(obs)>0:
                print("    HAS OBSERVATIONS, PLOTTING CALIBRATION")
                fig.add_trace(go.Scatter(x=obs['Month'], y=np.log(obs['Discharge']), name="Discharge Observations", mode='markers')) 
        
            fig.update_layout(title=gauge_name,xaxis_title='Stress Period',yaxis_title='Flows, Model Units (Ft^3/s), Log Normalized',legend_title='flow',  barmode='relative') #  yaxis_type='log'
            fig.write_html(export_directory+'Calib_'+str(clean_gauge_name)+'.html')
            fig.write_image(export_directory+'Calib_'+str(clean_gauge_name)+'.jpg', scale=5) # requires kaleido dependency (e.g. conda install kaleido)
        
        segment_rchdf.loc[:, 'GaugeName'] = gauge_name
        segment_rchdf.to_excel(export_directory+str(clean_gauge_name)+'.xlsx')
        
        CombinedSimulatedGaugeData = pd.concat([CombinedSimulatedGaugeData, segment_rchdf], ignore_index=True)
CombinedSimulatedGaugeData=pd.merge(CombinedSimulatedGaugeData,Observation_df,left_on=['GaugeName','Date'],right_on=['SiteName','Month'],how='left')       
CombinedSimulatedGaugeData['SimulatedLessObserved']=CombinedSimulatedGaugeData['Discharge']-CombinedSimulatedGaugeData['FLOW_INTO_RCH']
CombinedSimulatedGaugeData['LogObserved']=np.log(CombinedSimulatedGaugeData['Discharge']) 
CombinedSimulatedGaugeData['LogSimulated']=np.log(CombinedSimulatedGaugeData['FLOW_INTO_RCH']) 
CombinedSimulatedGaugeData['LogSimulatedLessObserved']=CombinedSimulatedGaugeData['LogObserved']-CombinedSimulatedGaugeData['LogSimulated'] 

CombinedSimulatedGaugeData.to_excel(export_directory+"AllGaugeData.xlsx")    



#==============================OTHER EXPLORATORY PLOTS IF YOU WANT THEM=========================
if OtherStatsAndPlots.lower()=='y':
    # print information for model-wide stats
    print("PRINTING MODEL WIDE INFO")
    fig = go.Figure()
    fig.add_trace(go.Bar( x=FullModelTimeseries['PERIOD'], y=FullModelTimeseries['FLOW_TO_AQUIFER'],name=str("FLOW_TO_AQUIFER")))          
    fig.add_trace(go.Bar( x=FullModelTimeseries['PERIOD'], y=FullModelTimeseries['FLOW_INTO_RCH'],name=str("FLOW_INTO_RCH"))) 
    fig.add_trace(go.Bar( x=FullModelTimeseries['PERIOD'], y=FullModelTimeseries['FLOW_OUT_OF_RCH'],name=str("FLOW_OUT_OF_RCH"))) 
        
    fig.update_layout(title='SFR Flows - ModelWide',xaxis_title='Stress Period',yaxis_title='Flows, Model Units (Ft^3/d)',legend_title='flow',barmode='stack')
    fig.write_html(export_directory+'SFR Flows - ModelWide.html')
    FullModelTimeseries.to_excel(export_directory+'SFR Flows - ModelWide.xlsx')
    
    # print information for By-segment Stats  #
    # print("PRINTING BY SEGMENT")
    # for segment in SegmentTimeseries['SEG'].unique():
    #     segmentdf=SegmentTimeseries[SegmentTimeseries['SEG']==segment]
        
    #     fig = go.Figure()
    #     fig.add_trace(go.Bar( x=segmentdf['PERIOD'], y=segmentdf['FLOW_TO_AQUIFER'],name=str("FLOW_TO_AQUIFER")))          
    #     #fig.add_trace(go.Bar( x=segmentdf['PERIOD'], y=segmentdf['FLOW_INTO_RCH'],name=str("FLOW_INTO_RCH"))) 
    #     #fig.add_trace(go.Bar( x=segmentdf['PERIOD'], y=segmentdf['FLOW_OUT_OF_RCH'],name=str("FLOW_OUT_OF_RCH"))) 
            
    #     fig.update_layout(title='SFR Flows Segment ' + str(segment),xaxis_title='WBS',yaxis_title='Flows, Model Units (Ft^3/d)',legend_title='flow',barmode='stack')
    #     fig.write_html(export_directory+'SFR Flows-Segment ' + str(segment)+'.html')
    #     segmentdf.to_excel(export_directory+'SFR Flows-Segment ' + str(segment)+'.xlsx')
    # SegmentTimeseries.to_excel(export_directory+'SFR Flows-AllSegments'+'.xlsx')
    
    print("PRINTING HEAD BY SEGMENT")
    fig = go.Figure()
    for segment in SegmentTimeseriesMean['SEG'].unique():
        segmentdf=SegmentTimeseriesMean[SegmentTimeseriesMean['SEG']==segment]
        
    
        fig.add_trace(go.Bar( x=segmentdf['PERIOD'], y=segmentdf['STREAM_HEAD'],name="SEG:"+str(segment)))          
        #fig.add_trace(go.Bar( x=segmentdf['PERIOD'], y=segmentdf['FLOW_INTO_RCH'],name=str("FLOW_INTO_RCH"))) 
        #fig.add_trace(go.Bar( x=segmentdf['PERIOD'], y=segmentdf['FLOW_OUT_OF_RCH'],name=str("FLOW_OUT_OF_RCH"))) 
            
    fig.update_layout(title='SFR STREAM_HEAD Flows-ALL SEGMENTS',xaxis_title='Stress Period',yaxis_title='STREAM_HEAD, Model Units (Ft)',legend_title='flow',barmode='stack')
    fig.write_html(export_directory+'SFR Flows STREAM_HEAD -ALL SEGMENTS'+'.html')
    segmentdf.to_excel(export_directory+'SFR ASTREAM_HEAD-ALL SEGMENTS'+'.xlsx')
    SegmentTimeseries.to_excel(export_directory+'SFR head-AllSegments'+'.xlsx')
    
    print("PRINTING AQUIFER FLOW BY SEGMENT")
    fig = go.Figure()
    for segment in SegmentTimeseries['SEG'].unique():
        segmentdf=SegmentTimeseries[SegmentTimeseries['SEG']==segment]
        
    
        fig.add_trace(go.Bar( x=segmentdf['PERIOD'], y=segmentdf['FLOW_TO_AQUIFER'],name="SEG:"+str(segment)))          
        #fig.add_trace(go.Bar( x=segmentdf['PERIOD'], y=segmentdf['FLOW_INTO_RCH'],name=str("FLOW_INTO_RCH"))) 
        #fig.add_trace(go.Bar( x=segmentdf['PERIOD'], y=segmentdf['FLOW_OUT_OF_RCH'],name=str("FLOW_OUT_OF_RCH"))) 
            
    fig.update_layout(title='SFR AQUIFER Flows-ALL SEGMENTS',xaxis_title='Stress Period',yaxis_title='TO_AQUIFER, Model Units (Ft^3/d)',legend_title='flow',barmode='stack')
    fig.write_html(export_directory+'SFR Flows AQUIFER -ALL SEGMENTS'+'.html')
    segmentdf.to_excel(export_directory+'SFR AQUIFERFlows-ALL SEGMENTS'+'.xlsx')
    SegmentTimeseries.to_excel(export_directory+'SFR AQUIFER Flows-AllSegments'+'.xlsx')
    
    print("PRINTING OUT FLOW BY SEGMENT")
    fig = go.Figure()
    for segment in SegmentTimeseries['SEG'].unique():
        segmentdf=SegmentTimeseries[SegmentTimeseries['SEG']==segment]
        
    
        fig.add_trace(go.Bar( x=segmentdf['PERIOD'], y=segmentdf['FLOW_OUT_OF_RCH'],name="SEG:"+str(segment)))          
        #fig.add_trace(go.Bar( x=segmentdf['PERIOD'], y=segmentdf['FLOW_INTO_RCH'],name=str("FLOW_INTO_RCH"))) 
        #fig.add_trace(go.Bar( x=segmentdf['PERIOD'], y=segmentdf['FLOW_OUT_OF_RCH'],name=str("FLOW_OUT_OF_RCH"))) 
            
    fig.update_layout(title='SFR Out Flows-ALL SEGMENTS',xaxis_title='Stress Period',yaxis_title='FLOW_OUT_OF_RCH, Model Units (Ft^3/d)',legend_title='flow',barmode='stack')
    fig.write_html(export_directory+'SFR Out Flows-ALL SEGMENTS'+'.html')
    segmentdf.to_excel(export_directory+'SFR Out Flows-ALL SEGMENTS'+'.xlsx')
    SegmentTimeseries.to_excel(export_directory+'SFR Out Flows-AllSegments'+'.xlsx')
    
    print("PRINTING IN FLOW BY SEGMENT")
    fig = go.Figure()
    for segment in SegmentTimeseries['SEG'].unique():
        segmentdf=SegmentTimeseries[SegmentTimeseries['SEG']==segment]
        
    
        #fig.add_trace(go.Bar( x=segmentdf['PERIOD'], y=segmentdf['FLOW_OUT_OF_RCH'],name="SEG:"+str(segment)))          
        fig.add_trace(go.Bar( x=segmentdf['PERIOD'], y=segmentdf['FLOW_INTO_RCH'],name=str("FLOW_INTO_RCH"))) 
        #fig.add_trace(go.Bar( x=segmentdf['PERIOD'], y=segmentdf['FLOW_OUT_OF_RCH'],name=str("FLOW_OUT_OF_RCH"))) 
            
    fig.update_layout(title='SFR In Flows-ALL SEGMENTS',xaxis_title='Stress Period',yaxis_title='FLOW_INTO_RCH, Model Units (Ft^3/d)',legend_title='flow',barmode='stack')
    fig.write_html(export_directory+'SFR In Flows-ALL SEGMENTS'+'.html')
    segmentdf.to_excel(export_directory+'SFR In Flows-ALL SEGMENTS'+'.xlsx')
    SegmentTimeseries.to_excel(export_directory+'SFR In Flows-AllSegments'+'.xlsx')


