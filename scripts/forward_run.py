import os
import numpy as np
import pandas as pd
import pyemu
import multiprocessing as mp
import shutil
import flopy

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


def write_pilot_point(layer, prop, model_ws):
    if layer != 1:
        factors_file = os.path.join(model_ws, 'pp2024', "pp.fac")
    else:
        factors_file = os.path.join(model_ws, 'pp2024', "pp2.fac")

    out_file = os.path.join(model_ws, 'pp2024_out', f"{prop}.txt")

    pp_file = os.path.join(model_ws, 'pp2024', f"{prop}pp.dat")
    assert os.path.exists(pp_file), f"pp_file does not exist {pp_file}"
    print(f"pp_file = {pp_file}, factors_file={factors_file}, out_file={out_file}")

    hk_arr = pyemu.geostats.fac2real(pp_file, factors_file=factors_file, out_file=out_file)


def write_all_pp(model_ws):
    '''write outputs from pilots points, creating actual grid files'''
    prefix_dict = {0: ["hk1", "sy1", "vk1"],
                   1: ["hk2", "ss2", "vk2", 'fmp_vk', 'drn_k'],
                   2: ["hk3", "ss3", "vk3"],
                   3: ["hk4", "ss4", "vk4"],
                   4: ["hk5", "ss5", "vk5"],
                   5: ["hk6", "ss6", "vk6"]}
    for lay in prefix_dict.keys():
        for par in prefix_dict[lay]:
            write_pilot_point(lay, par, model_ws)


def summarize_budget(folder):
    '''summarize the budget output terms for cleaner ingetion into pest processing'''
    b = os.path.join(folder, 'output', "Budget.txt")
    bud = pd.read_csv(b, sep='\s+')

    bud = bud.groupby(pd.to_datetime(bud.DATE_START).dt.year).sum()

    bud = bud.loc[:, ['STORAGE_IN', 'STORAGE_OUT',
                      'WEL_V1_OUT', 'GHB_IN', 'RCH_IN', 'DRT_OUT', 'SFR_IN', 'SFR_OUT',
                      'MNW2_IN', 'MNW2_OUT', 'FMP_WELLS_OUT', 'FMP_FNR_IN', 'FMP_FNR_OUT',
                      'IN_OUT', 'PERCENT_ERROR'], ]

    bud.to_csv(b.replace('Budget', 'Budget_pest'))
    bud = bud.sum().to_frame('sum')
    bud.index.name = 'flux_term'
    bud.insert(0, 'ref', 1)
    bud.to_csv(b.replace('Budget', 'Budget_pest_summary'))
    return bud


def sfr_flows_log_transform(infile, outfile):
    ofile_sep = '\s+'
    ofile_skip = [0]
    names = "Time           Stage            Flow     Conductance        HeadDiff       Hyd.Grad.".split()
    q = pd.read_csv(infile, sep=ofile_sep, skiprows=ofile_skip, names=names)
    q.to_csv(outfile)
    c = q.Flow < 0.001 * 60 * 60 * 24  # all flows below this threshold will be replaced to this value (USGS does not report below)
    q.loc[c, 'Flow'] = 0.001 * 60 * 60 * 24  # ft^3/day
    q.loc[:, "Flow"] = q.Flow.apply(np.log10)
    q.to_csv(outfile.replace(".csv", '_log.csv'))
    return q


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

def run_all_hyd_obs(workspace):
    bigobj = load_hydobs(workspace)
    abs_small = down_sample_hydobs(bigobj, numper_diff = 5,keep_every = 4)
    abs_obs = create_obs_from_hyd(abs_small)

    print(f'there are {abs_obs.shape[0]} observations in the GWLE observation absolute elev file')

    diff_big = create_diff_for_hydobs(bigobj, numper_diff = 5)
    diff_small = down_sample_hydobs(diff_big, numper_diff = 5,keep_every = 4)
    diff_obs = create_obs_from_hyd(diff_small)

    print(f'there are {diff_obs.shape[0]} observations in the GWLE observation drawdown elev file')
    
    f_abs = os.path.join(os.path.join(workspace, "GWLE_OBS", 'gwle_asbolute_mod_heads.csv'))
    print(f"writing absolute heads to {f_abs}")
    f_diff = os.path.join(os.path.join(workspace, "GWLE_OBS", 'gwle_drawdown_mod_heads.csv'))
    print(f"writing drawdon heads to {f_diff}")

    abs_obs.to_csv(f_abs)
    diff_obs.to_csv(f_diff)

    print("Done writing to files")

if __name__ == '__main__':
    mp.freeze_support()
    foldr = os.getcwd()
    # Usage
    directory_path = os.path.join(foldr, 'pp2024_out')
    delete_all_files_in_directory(directory_path)

    print(os.getcwd())
    write_all_pp(foldr)
    read_drain(foldr)

    if pyemu.os_utils.platform.system() == 'Windows':
        print('running with windows executable')
        pyemu.os_utils.run(r'mf-owhm.exe SVIGFM_GSP.nam')
    else:
        print('running with linux executable')
        pyemu.os_utils.run(r'mf-owhm.nix SVIGFM_GSP.nam')

    summarize_budget(foldr)
    _ = sfr_flows_log_transform(os.path.join(foldr, 'output', "kenwood_sfr.dat"),
                                os.path.join(foldr, 'output', "kenwood_sfr_reformat.csv"))

    _ = sfr_flows_log_transform(os.path.join(foldr, 'output', "agua_caliente_sfr.dat"),
                                os.path.join(foldr, 'output', "agua_caliente_sfr_reformat.csv"))

    run_all_hyd_obs(foldr)
