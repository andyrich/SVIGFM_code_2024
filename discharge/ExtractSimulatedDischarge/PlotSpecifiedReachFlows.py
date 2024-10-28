# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:25:13 2024

    @author: pwickham
    Script Name: PlotSpecifiedReachFlows.py
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
        Parameters:
    Times (list): List of two strings representing start and end dates.
    htmlPlots (str): Flag to generate HTML plots ('y' or 'n').
    CalibPlots (str): Flag to generate calibration plots ('y' or 'n').
    GaugeLocations (dict): Dictionary of gauge locations with names as keys and ['row_col'] format as values.

        
    DEVELOPMENT NOTES
        `Currently set up to extract and process information, and plot htmls and jpgs.
        `Added earlier observation data for USGS gauges like kenwood, agua caliente, and Nathanson
        `note '0' values will get dropped when logged
        `Made the following changes after call on 2024/10/25:
                ` print out all results as csv
                ` print out simplified results
                `make times in python rather than relying on input file
                `made into a callable function so that Andy can use it with pest 
                `made pathing suitible for linux
        

"""
def ProcessSFRForPest(Times=['12/1/1969','9/1/2018'],htmlPlots='n',CalibPlots='n', GaugeLocations={ # dictionary of gauge locations in format 'Name': ['row_col']
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
                }):
    import os
    export_directory=os.getcwd()  #Where to post output. currently set as script location 
    File2Process=os.path.join(export_directory,'output','SV_sfr_sim.dat') # output filw
    ObservationData=os.path.join(export_directory,'CombinedDischargeObservationData.xlsx') # observation data in database format
    import pandas as pd    
    import re
    #import os
    if htmlPlots.lower()=='y' or CalibPlots.lower()=='y':
        import plotly.graph_objects as go
#    import winsound
    import numpy as np
    from datetime import datetime
    #import matplotlib
    #import matplotlib.pyplot as plt
    #import numpy as np
    #import time
    #import sys 
    
    #==================================DEFINE FUNCTIONS=========================
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
    
    #FullModelTimeseries=  df.groupby(['PERIOD'],as_index=False).sum(numeric_only=True) # group the whole model (total flow)
    #SegmentTimeseries=  df.groupby(['SEG','PERIOD'],as_index=False).sum(numeric_only=True) # group by segment (total flow)
    #SegmentTimeseriesMean=df.groupby(['SEG','PERIOD'],as_index=False).mean(numeric_only=True)  # group by segment (mean flow)
    ReachTimeseries=  df.groupby(['SEG','RCH','PERIOD'],as_index=False).mean(numeric_only=True) # group by reach (mean flow) -- note that this row shouldn't practically introduce any change from df to ReachTimeseries. But it does make a copy. 
    ReachTimeseries['SEG_RCH']=ReachTimeseries['SEG'].astype(int).astype(str)+"_"+ReachTimeseries['RCH'].astype(int).astype(str) # unique ID for reach
    ReachTimeseries['ROW_COL']=ReachTimeseries['ROW'].astype(int).astype(str)+"_"+ReachTimeseries['COL'].astype(int).astype(str) # unique ID for reach by row/col
    
    start_date = datetime.strptime(Times[0], '%m/%d/%Y') # make datetime
    end_date = datetime.strptime(Times[1], '%m/%d/%Y')
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS') # range in monthly frequency
    Dates_DF = pd.DataFrame({'Date': date_range,'DaysInMonth': date_range.days_in_month}) # make dates dataframe akin to previous excel one
    Dates_DF['sp']=Dates_DF.index+1
    ReachTimeseries=pd.merge(ReachTimeseries,Dates_DF,left_on='PERIOD',right_on='sp')
    Columns2Convert=['FLOW_INTO_RCH', 'FLOW_TO_AQUIFER', 'FLOW_OUT_OF_RCH', 
                    'OVRLND_FLOW', 'DIRECT_PRECIP', 'STREAM_ET']
    for column in Columns2Convert:    ReachTimeseries[column] = ReachTimeseries[column] / (60*60*24) # convert to cfs
    
    # read observations
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
            
                fig.update_layout(title=gauge_name,xaxis_title='Stress Period',yaxis_title='Flows, Model Units (Ft^3/s)',legend_title='flow',  barmode='relative') #  yaxis_type='log'
                fig.write_html(export_directory+'Calib_'+str(clean_gauge_name)+'.html')
                try:
                    fig.write_image(export_directory+'Calib_'+str(clean_gauge_name)+'.jpg', scale=5) # requires kaleido dependency (e.g. conda install kaleido)
                except:
                    print("       FOR PRINTING JPGS, INSTALL KALEIDO")
            segment_rchdf.loc[:, 'GaugeName'] = gauge_name
            segment_rchdf.to_excel(export_directory+str(clean_gauge_name)+'.xlsx')
            
            CombinedSimulatedGaugeData = pd.concat([CombinedSimulatedGaugeData, segment_rchdf], ignore_index=True)
    CombinedSimulatedGaugeData=pd.merge(CombinedSimulatedGaugeData,Observation_df,left_on=['GaugeName','Date'],right_on=['SiteName','Month'],how='left')       
    
    CombinedSimulatedGaugeData['SimulatedLessObserved']=CombinedSimulatedGaugeData['FLOW_INTO_RCH']-CombinedSimulatedGaugeData['Discharge']
    CombinedSimulatedGaugeData['LogObserved']=np.where(CombinedSimulatedGaugeData['Discharge'] != 0,np.log(CombinedSimulatedGaugeData['Discharge']),np.nan) # handle log transformation on zero values by setting to nan. can set to zero if preferred
    CombinedSimulatedGaugeData['LogSimulated'] = np.where(CombinedSimulatedGaugeData['FLOW_INTO_RCH'] != 0,np.log(CombinedSimulatedGaugeData['FLOW_INTO_RCH']),np.nan) # handle log transformation on zero values by setting to nan. can set to zero if preferred
    CombinedSimulatedGaugeData['LogSimulatedLessObserved']=CombinedSimulatedGaugeData['LogSimulated']-CombinedSimulatedGaugeData['LogObserved']
    
    CombinedSimulatedGaugeData.to_csv(export_directory+"AllGaugeData.csv")    
    
    #simplify for PEST
    Results4Pest=pd.DataFrame(CombinedSimulatedGaugeData[['GaugeName','SEG','RCH','LAYER','ROW','COL','DaysInMonth']])
    Results4Pest['Date']=CombinedSimulatedGaugeData['Date_x']
    Results4Pest['Simulated']=CombinedSimulatedGaugeData['FLOW_INTO_RCH']
    Results4Pest['Observed']=CombinedSimulatedGaugeData['Discharge']
    Results4Pest['SimulatedLessObserved']=CombinedSimulatedGaugeData['SimulatedLessObserved']
    Results4Pest=Results4Pest.dropna() # drop where there is no observed data for PEST to match
    CombinedSimulatedGaugeData.to_csv(export_directory+"GaugeData4Pest.csv")    

ProcessSFRForPest()


