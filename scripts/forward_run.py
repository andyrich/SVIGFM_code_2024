import os
import numpy as np
import pandas as pd
import pyemu
import multiprocessing as mp
import shutil
import flopy

def get_zone_bounds():
    '''multiplier bounds for pilot points and zones'''
    zone_bounds =   dict(  sy1 = [0.1, 100], #lo, hi
                ss1 = [0.001  ,100],
                ss2 = [0.001  ,100],
                ss3 = [0.001  ,100],
                ss4 = [0.001  ,100],
                ss5 = [0.001  ,100],
                ss6 = [0.001  ,100],
                vk1 = [0.001  ,100],
                vk2 = [0.001  ,100],
                vk3 = [0.001  ,100],
                vk4 = [0.001  ,100],
                vk5 = [0.001  ,100],
                vk6 = [0.001  ,100],
                hk1 = [0.001  ,100],
                hk2 = [0.001  ,100],
                hk3 = [0.001  ,100],
                hk4 = [0.001  ,100],
                hk5 = [0.001  ,100],
                hk6 = [0.001  ,100],
                drn_k = [0.001, 100],
             fmp_vk = [0.0001, 100],   )

    return zone_bounds

def get_parbounds():
    '''parameter bounds for OTHER parameters'''
    parbounds = dict(sfr = [0.0001, 1000, 1e-2], #lo, hi, initial values
        fieswp = [0.5, 0.999, 0.7],

        hfb= [0.000001, 1000, 1000],
        fmp_kc= [0.5, 2, 1.4], #individual kc multipliers
        fmp_ofe= [0.5, 1.0, 0.7,], #individual OFE values
        fmp_sfac = [.5,2,1.4],

        rurfac = [.8,1.25,1.0],

        laymult_drn_k = [10, 5000, 1200],
        laymult_fmp_vk = [0.0001, .01, 0.001],
        ghbk = [0.0001, 10000, 1.4E-02],
        laymult_hk =[1e-5, 1000, 1.],
        laymult_vk = [1e-3, 1e-1, 1.],
        laymult_ss = [1e-6, 1e-3, 1.],
        laymult_sy = [0.0001, 0.3, 1.]
        ) # all sfac/kc multipliers
    
    return parbounds

def get_bounds():
    '''actual paremter bounds to be enforced at time of writing arrays'''
    bounds = dict(sy1 = [0.0001, 0.3], #lo, hi
                    ss1 = [1e-6, 0.3/50],
                    ss2 = [1e-6, 1e-3],
                    ss3 = [1e-6, 1e-3],
                    ss4 = [1e-6, 1e-3],
                    ss5 = [1e-6, 1e-3],
                    ss6 = [1e-6, 1e-3],
                    vk1 = [1e-3, 1e-1],
                    vk2 = [1e-3, 1e-1],
                    vk3 = [1e-3, 1e-1],
                    vk4 = [1e-3, 1e-1],
                    vk5 = [1e-3, 1e-1],
                    vk6 = [1e-3, 1e-1],
                    hk1 = [1e-5, 1000],
                    hk2 = [1e-5, 1000],
                    hk3 = [1e-5, 1000],
                    hk4 = [1e-5, 1000],
                    hk5 = [1e-5, 1000],
                    hk6 = [1e-5, 1000],
                    drn_k = [10, 50000],
                    fmp_vk = [0.0001, .01],
                     )
    return bounds

def get_prefix_dict_for_pilot_points():
    prefix_dict = {0: ["hk1", 'ss1', "sy1", "vk1"],
               1: ["hk2", "ss2", "vk2", 'fmp_vk', 'drn_k'],
               2: ["hk3", "ss3", "vk3"],
               3: ["hk4", "ss4", "vk4"],
               4: ["hk5", "ss5", "vk5"],
               5: ["hk6", "ss6", "vk6"]}

    return prefix_dict

def read_drain(folder):
    # Step 1: Open the text file and read lines
    f = os.path.join(folder, 'drt.drt')
    with open(f, 'r') as file:
        lines = file.readlines()

    # Step 2: Extract the number n from the 4th line after splitting by spaces
    n = int(lines[8].split()[0])

    # Step 3: Read the next n rows from a CSV file using pandas
    # Assuming the CSV file is named 'data.csv' and is in the same directory
    csv_data = pd.read_csv(f, skiprows=9, nrows=n, sep='\s+',
                           names=['lay', 'i', 'j', 'top', 'k', 'd', 'farm', 'd2', 'd3'])

    g = os.path.join(folder, 'pp2024_out', 'drn_k.txt')
    # g = os.path.join(folder,'model_arrays', 'uzf_vk.uzf'  )
    # vk = np.genfromtxt(g,delimiter = ',')
    vk = np.genfromtxt(g)
    csv_data.loc[:, 'k'] = vk[csv_data.loc[:, 'i'] - 1, csv_data.loc[:, 'j'] - 1]
    # Display the data read from CSV

    head = '''#Drain Returnflow
    BEGIN OPTIONS
         RETURNFLOW
    #    PRINTFILE  89
    #    DBFILE    ./Output/DRT_Output.txt
         NOPRINT
    END OPTIONS
    '''
    head2 = "{} {} 0 0\n{} {}\n".format(csv_data.shape[0], 50, csv_data.shape[0], 0)

    csv_data = csv_data.loc[:, ['lay', 'i', 'j', 'top', 'k', 'd', 'farm', 'd2', 'd3']]

    with open(os.path.join(folder, 'drt.drt'), 'w') as w:
        w.write(head)
        w.write(head2)
        csv_data.to_csv(w, lineterminator='\n', index=False, header=False, sep='\t')
        for i in range(800):
            w.write('-1\n')

def write_kc(folder):
    ##read multiplier dict as 
    multiplier_dict = pd.read_csv(os.path.join(folder, 'kc_scale_factors.csv' ), index_col = 0)
    multiplier_dict.index = multiplier_dict.index.str.strip()
    multiplier_dict = multiplier_dict.loc[:,'parvalue'].to_dict()
    kcfile_in = os.path.join(folder, 'fmp_input', 'model_arrays', 'kc_GSP.txt')
    kcfile_out = os.path.join(folder, 'fmp_input', 'model_arrays', 'kc_GSP_PEST.txt')
    SFAC = multiplier_dict['SFAC']
    # Open the input file and prepare the output file
    with open(kcfile_in, 'r') as infile, open(kcfile_out, 'w') as outfile:
        for line in infile:
            parts = line.split()
            
            # Check if the line starts a new section
            if "SFAC" in line:
                # current_multiplier = multiplier_dict[parts[1]]
                line = f"{parts[0]}\t{SFAC}\t{parts[2]}\t{parts[3]}\n"
                outfile.write(line)  # Write the header line as-is
            else:
                # Modify the second column
                # try:
                index = parts[0]
                value = float(parts[1])
                other = parts[2]

                # Apply the multiplier to the value
                modified_value = value * multiplier_dict[index]

                # Write the modified line to the output file
                outfile.write(f"{index}\t{modified_value:.4f}\t{other}\n")
                
    print(f'done writing kc file to:\n\t{kcfile_out}')


def write_OFE(folder):
    '''read the on-farm efficiencies (OFE) to create the repeated values of inputs'''
    ofe = os.path.join(folder,  'ofe_scale_factors.csv' )
    print(f'reading {ofe} for OFE factors')
    multiplier_dict = pd.read_csv(ofe, index_col = 0)
    vals = multiplier_dict.parvalue.values

    nfarms = 139
    # Create a DataFrame by repeating the values and adjusting for the number of rows
    df = pd.DataFrame([vals] * nfarms, index = np.arange(1, nfarms+1))

    outfile = os.path.join(folder, 'fmp_update','sv_OFE_GSP.txt')
    with open(outfile, 'w')  as out:
        for date in pd.date_range('12/1/1969', freq = 'ME', end = '1/1/2026'):
            out.write(f"SFAC  1.0   {date.strftime('%Y-%b')}\n")
            # for i in range(nfarms):
            #     out.write(vals)
            # df.to_csv(out, mode = 'append')
            out.write(df.to_string(index=True, header = False ))
            out.write('\n')  # Optional: add a newline after the DataFrame

    print(f'done writing OFE to {outfile}')

def write_pilot_point(layer, prop, model_ws):
    if layer != 1:
        factors_file = os.path.join(model_ws, 'pp2024', "pp.fac")
    else:
        factors_file = os.path.join(model_ws, 'pp2024', "pp2.fac")

    # out_file = os.path.join(model_ws, 'pp2024_out', f"{prop}.txt")
    out_file = None

    pp_file = os.path.join(model_ws, 'pp2024', f"{prop}pp.dat")
    assert os.path.exists(pp_file), f"pp_file does not exist {pp_file}"
    print(f"pp_file = {pp_file}, factors_file={factors_file}, out_file={out_file}")

    #write the pilot points file here, but it is not actually used (just using the in-memory array file instead, below)
    hk_arr = pyemu.geostats.fac2real(pp_file, factors_file=factors_file, out_file=out_file)

    # load the zone multiplier and layer multiplier
    mul_f = os.path.join(model_ws, 'zone_pest_mult', f"zonemult_{prop}.csv")
    assert os.path.exists(mul_f), f"{mul_f} does not exist"
    mult = np.genfromtxt(mul_f,delimiter=',')

    lay_f = os.path.join(model_ws, 'zone_pest_mult', f"laymult_{prop}.csv")
    assert os.path.exists(lay_f), f"{lay_f} does not exist"
    lay = np.genfromtxt(lay_f,delimiter=',')

    assert mult.shape==lay.shape==hk_arr.shape, f"shapes are not equal\nmult.shape=={mult.shape}\nlay.shape=={lay.shape}\nhk_arr.shape=={hk_arr.shape}\n"

    out = hk_arr*mult*lay

    #set parameter bounds
    bounds = get_bounds()[prop]
    out[out<bounds[0]] = bounds[0]
    out[out>bounds[1]] = bounds[1]
    #final values:
    array_out = os.path.join(model_ws, 'pp2024_out', f"{prop}.txt")
    np.savetxt(array_out, out)

def write_all_pp(model_ws):
    '''write outputs from pilots points, creating actual grid files'''

    prefix_dict = get_prefix_dict_for_pilot_points()
    
    for lay in prefix_dict.keys():
        for par in prefix_dict[lay]:
            write_pilot_point(lay, par, model_ws)


def summarize_budget(folder):
    '''summarize the budget output terms for cleaner ingetion into pest processing'''
    b = os.path.join(folder, 'output', "Budget.txt")
    bud = pd.read_csv(b, sep='\s+')

    listx = ['PERCENT_ERROR']
    # Define aggregation functions for each column
    agg_funcs = {col: ('mean' if col in listx else 'sum') for col in bud.columns}
    
    # Perform the groupby and aggregation
    bud = bud.groupby(pd.to_datetime(bud.DATE_START).dt.year).agg(agg_funcs)
    
    
    # bud = bud.groupby(pd.to_datetime(bud.DATE_START).dt.year).sum()

    bud = bud.loc[:, ['STORAGE_IN', 'STORAGE_OUT', "DRT_OUT",
                      'RURWELLS_OUT', "MNIWELLS_OUT", 'GHB_IN', 'RCH_IN', 'SFR_IN', 'SFR_OUT',
                      'MNW2_IN', 'MNW2_OUT', 'FMP_WELLS_OUT', 'FMP_FNR_IN', 'FMP_FNR_OUT',
                      'IN_OUT', 'PERCENT_ERROR'], ]
    
    bud.index = pd.to_datetime(bud.index, format = '%Y')
    bud.index.name = 'Date'
    bud.to_csv(b.replace('Budget', 'Budget_pest'))
    # bud = bud.sum().to_frame('sum')
    # Define aggregation functions for each column
    agg_funcs = {col: ('mean' if col in listx else 'sum') for col in bud.columns}
    bud = bud.agg(agg_funcs).to_frame('summary')
    bud.index.name = 'flux_term'
    bud.insert(0, 'ref', 1)
    bud.to_csv(b.replace('Budget', 'Budget_pest_summary'))
    return bud


def sfr_flows_log_transform(infile, outfile, station):
    ofile_sep = '\s+'
    ofile_skip = [0]
    names = "Time           Stage            Flow     Conductance        HeadDiff       Hyd.Grad.".split()
    q = pd.read_csv(infile, sep=ofile_sep, skiprows=ofile_skip, names=names)
    q.loc[:,'Station'] = station
    q.loc[:,'Date'] = (pd.to_timedelta(q.Time.astype(float), unit = "D")+pd.to_datetime('11/30/1969'))
    q = q.astype({'Date':'datetime64[ns]'})

    q.to_csv(outfile)
    c = q.Flow < 0.001 * 60 * 60 * 24  # all flows below this threshold will be replaced to this value (USGS does not report below)
    q.loc[c, 'Flow'] = 0.001 * 60 * 60 * 24  # ft^3/day
    q.loc[:, "Flow"] = q.Flow.apply(np.log10)
     
    q.to_csv(outfile.replace(".csv", '_log.csv'))
    return q


def check_obs_flows(folder, station):
    '''for some reason the observed flows are getting erased. this just checks if it has water year in it'''

    outfile = os.path.join(folder,'output',f'sfr_{station}_modeled_fdc_curve.csv')
    outfile_min = os.path.join(folder,'output',f'sfr_{station}_modeled_water_year_min.csv')
    
    obs_flow = pd.read_csv(os.path.join(folder,f'sfr_{station}_obs.csv'), index_col = 0)

    if not 'Water Year' in obs_flow.columns:
        raise ValueError(f"There is no water year column in {os.path.join(folder,f'sfr_{station}_obs.csv')}")

    print('\npassed water year check for obs file\n')
    print(os.path.join(folder,f'sfr_{station}_obs.csv'))


def sfr_flow_accum(folder, station):
    '''load the observed flow to get the dates that the USGS gauge is active. must have the observed sfr file in the main dir'''
    outfile = os.path.join(folder,'output',f'sfr_{station}_modeled_fdc_curve.csv')
    outfile_min = os.path.join(folder,'output',f'sfr_{station}_modeled_water_year_min.csv')

    print(f'\nprinting the fdc sfr for {station} to:\n{outfile}')
    print(f'printing the water min flow for sfr for {station} to:\n{outfile_min}')
    
    obs_flow = pd.read_csv(os.path.join(folder,f'sfr_{station}_obs.csv'), index_col = 0)
    obs_flow = obs_flow.astype({'date':'datetime64[ns]','Water Year':'datetime64[ns]'}).drop(columns = 'Q')

    print(f"loading sfr for {os.path.join(folder,f'sfr_{station}_obs.csv')}\n")

    
    sfr = os.path.join(folder, 'output', f"{station}_sfr_reformat.csv")
    flow = pd.read_csv(sfr, index_col = 0)
    flow = flow.astype({'Date':'datetime64[ns]'})

    
    #fix low values for modeled data
    c = flow.Flow < 0.001 * 60 * 60 * 24  # all flows below this threshold will be replaced to this value (USGS does not report below)
    flow.loc[c, 'Flow'] = 0.001 * 60 * 60 * 24  # ft^3/day


    flow = pd.merge(obs_flow, flow, left_on = 'date', right_on = 'Date')

    
    
    fdf = {x:np.percentile(flow.Flow.values, x) for x in range(5,101,5)}
    fdf = pd.DataFrame.from_dict(fdf, orient = 'index', columns = ['term'] )
    fdf.loc[:,'term'] = fdf.term.apply(np.log10)
    fdf.loc[:,'Station'] = station
    fdf.loc[:,'desc'] = 'flow duration curve value'
    fdf.index.name = 'fdc_percentile'

    fdf.to_csv(outfile)
    print(fdf.head().loc[:,['term','Station','desc']])

    wymin = flow.groupby('Water Year').min()
    wymin = wymin.drop(columns = ['Date','date'])

    wymin.loc[:,'term'] = wymin.Flow.apply(np.log10)
    wymin.loc[:,'Date'] = wymin.index
    wymin.loc[:,'Station'] = station
    wymin.loc[:,'desc'] = 'yearly_min_value'
    wymin.index.name = 'yearly_min_value'
    wymin.to_csv(outfile_min)
    print(wymin.head().loc[:,['Date','term','Station','desc']])

    return fdf, wymin

def delete_all_files_in_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    # Loop through all the files and directories in the specified directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            # Check if it's a file or directory and remove it accordingly
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    print(f"\nAll files in the directory {directory_path} have been deleted.\n")

def load_hydobs(workspace):
    '''creating super large file of heads observations'''
    out = flopy.utils.HydmodObs(os.path.join(workspace, 'output', 'SV_hyd.hyd'),)
    out = out.get_dataframe(start_datetime='12/1/1969')
    out = out.rename(columns = lambda x: x[6:]  if len(x)>12 else x)
    out = out.drop(columns = 'totim')

    out = out.reindex(pd.date_range('1975-01-01', freq = 'MS', periods = 525))

    return out

def create_diff_for_hydobs(df, numper_diff = 5):
    '''will take the drawdown of very (numper_diff) nth measurement (the zeroth and every numper_diff will be used as drawdown head)'''
    df = df.apply(calculate_differences, n=numper_diff)

    return df

def down_sample_hydobs(df, numper_diff = 5,keep_every = 4):
    "remove numper_diff measurements. then it will keep keep_every nth value"
    df = df.drop(df.iloc[::numper_diff, :].index)
    df = df.iloc[::keep_every, :]

    return df

def create_obs_from_hyd(sim):
    '''converting super large file of heads observations from wide to long to be loaded via pestpp'''
    sim = sim.stack()
    sim.index = sim.index.set_names([ 'date','Station',])
    sim = sim.swaplevel().to_frame('meas').reset_index()

    return sim


def calculate_differences(series, n):
    '''function to create differences at every n for the hydmobs datasets'''
    differences = []
    for i in range(0, len(series), n):
        chunk = series[i:i+n]
        first_value = chunk[0]
        diff = [value - first_value for value in chunk]
        differences.extend(diff)
    return differences

def rolling_mean(df, nyears = 10):
    '''rolling mean of hydobs/and gwle data. dates are labeled as end of period. for 10-year it is past end of model, but just represents periods>2015-12-31'''
    df = df.resample(f'{nyears}Y').mean()

    return df

def run_all_hyd_obs(workspace):
    bigobj = load_hydobs(workspace)
    abs_small = down_sample_hydobs(bigobj, numper_diff = 5,keep_every = 4)
    abs_obs = create_obs_from_hyd(abs_small)

    print(f'there are {abs_obs.shape[0]} observations in the GWLE observation absolute elev file')

    diff_big = create_diff_for_hydobs(bigobj, numper_diff = 5)
    diff_small = down_sample_hydobs(diff_big, numper_diff = 5,keep_every = 4)
    diff_obs = create_obs_from_hyd(diff_small)

    print(f'there are {diff_obs.shape[0]} observations in the GWLE observation drawdown elev file')

    # rolling observations
    roll = rolling_mean(bigobj,nyears=10)
    roll_obs = create_obs_from_hyd(roll)

    f_abs = os.path.join(os.path.join(workspace, "GWLE_OBS", 'gwle_asbolute_mod_heads.csv'))
    print(f"writing absolute heads to {f_abs}")
    f_diff = os.path.join(os.path.join(workspace, "GWLE_OBS", 'gwle_drawdown_mod_heads.csv'))
    print(f"writing drawdon heads to {f_diff}")
    f_roll = os.path.join(os.path.join(workspace, "GWLE_OBS", 'gwle_rolling_mod_heads.csv'))
    print(f"writing rolling heads to {f_roll}")

    abs_obs.to_csv(f_abs)
    diff_obs.to_csv(f_diff)
    roll_obs.to_csv(f_roll)

    print("Done writing to files")


def post_process(folder):
    check_obs_flows(folder, 'kenwood')
    check_obs_flows(folder, 'aguacal')
    
    summarize_budget(folder)
    _ = sfr_flows_log_transform(os.path.join(folder, 'output', "kenwood_sfr.dat"),
                                os.path.join(folder, 'output', "kenwood_sfr_reformat.csv"),
                                station = 'kenwood')

    _ = sfr_flows_log_transform(os.path.join(folder, 'output', "agua_caliente_sfr.dat"),
                                os.path.join(folder, 'output', "aguacal_sfr_reformat.csv"),
                               station = 'aguacal')

    run_all_hyd_obs(folder)

    _=sfr_flow_accum(folder,'kenwood')
    _=sfr_flow_accum(folder,'aguacal')

    
if __name__ == '__main__':
    mp.freeze_support()
    foldr = os.getcwd()
    # Usage
    directory_path = os.path.join(foldr, 'pp2024_out')
    delete_all_files_in_directory(directory_path)

    print(os.getcwd())
    write_all_pp(foldr)
    read_drain(foldr)

    write_kc(foldr)
    write_OFE(foldr)
    check_obs_flows(foldr, 'kenwood')
    check_obs_flows(foldr, 'aguacal')
    
    if pyemu.os_utils.platform.system() == 'Windows':
        print('running with windows executable')
        pyemu.os_utils.run(r'mf-owhm.exe SVIGFM_GSP.nam')
    else:
        print('running with linux executable')
        pyemu.os_utils.run(r'mf-owhm.nix SVIGFM_GSP.nam')

    post_process(foldr)
