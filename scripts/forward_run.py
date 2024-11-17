import os
import warnings

import numpy as np
import pandas as pd
import pyemu
import multiprocessing as mp
import shutil
import flopy
import flopy.utils.binaryfile as bf
from datetime import datetime
from scipy.stats import hmean

# import sfr_processing
print(f"loading forward_run.py from {os.getcwd()}")


def set_zoneentire_start_values(df):
    '''set partrans and parval1 in the parameter data df'''

    assert isinstance(df, pd.DataFrame)

    parvals = dict(
        # ['partrans', 'parval1', 'parlbnd', 'parubnd']
        zoneentire_drn_k_aguacal=['log', 1, 0.001, 1000],
        zoneentire_drn_k_bay=['log', 1, 0.001, 1000],
        zoneentire_drn_k_cityson=['log', 1, 0.001, 1000],
        zoneentire_drn_k_eastside=['log', 1, 0.001, 1000],
        zoneentire_drn_k_highlands=['log', 1, 0.001, 1000],
        zoneentire_drn_k_kenwood=['log', 1, 0.001, 1000],
        zoneentire_drn_k_southcent=['log', 1, 0.001, 1000],
        zoneentire_drn_k_vom=['log', 1, 0.001, 1000],
        zoneentire_drn_k_westside=['log', 1, 0.001, 1000],
        zoneentire_fmp_vk_aguacal=['log', 1, 0.001, 1000],
        zoneentire_fmp_vk_bay=['log', 1, 0.001, 1000],
        zoneentire_fmp_vk_cityson=['log', 1, 0.001, 1000],
        zoneentire_fmp_vk_eastside=['log', 1, 0.001, 1000],
        zoneentire_fmp_vk_highlands=['log', 1, 0.001, 1000],
        zoneentire_fmp_vk_kenwood=['log', 1, 0.001, 1000],
        zoneentire_fmp_vk_southcent=['log', 1, 0.001, 1000],
        zoneentire_fmp_vk_vom=['log', 1, 0.001, 1000],
        zoneentire_fmp_vk_westside=['log', 1, 0.001, 1000],
        zoneentire_hk_aguacal=['log', 1, 0.001, 1000],
        zoneentire_hk_bay=['log', 1, 0.001, 1000],
        zoneentire_hk_cityson=['log', 0.1, 0.001, 1000],  # not equal to 1
        zoneentire_hk_eastside=['log', 0.001, 0.001, 1000],  # not equal to 1
        zoneentire_hk_highlands=['log', 0.1, 0.001, 1000],  # not equal to 1
        zoneentire_hk_kenwood=['log', 1, 0.001, 1000],
        zoneentire_hk_southcent=['log', 1.0, 0.001, 1000],
        zoneentire_hk_vom=['log', 1, 0.001, 1000],
        zoneentire_hk_westside=['log', 0.1, 0.001, 1000],  # not equal to 1
        zoneentire_ss_aguacal=['log', 1, 0.001, 1000],
        zoneentire_ss_bay=['log', 1, 0.001, 1000],
        zoneentire_ss_cityson=['log', 1, 0.001, 1000],
        zoneentire_ss_eastside=['log', 0.1, 0.001, 1000],  # not equal to 1
        zoneentire_ss_highlands=['log', 1, 0.001, 1000],
        zoneentire_ss_kenwood=['log', 1, 0.001, 1000],
        zoneentire_ss_southcent=['log', 1, 0.001, 1000],
        zoneentire_ss_vom=['log', 1, 0.001, 1000],
        zoneentire_ss_westside=['log', 1, 0.001, 1000],
        zoneentire_sy_aguacal=['log', 1, 0.001, 1000],
        zoneentire_sy_bay=['log', 1, 0.001, 1000],
        zoneentire_sy_cityson=['log', 1, 0.001, 1000],
        zoneentire_sy_eastside=['log', 1, 0.001, 1000],
        zoneentire_sy_highlands=['log', 1, 0.001, 1000],
        zoneentire_sy_kenwood=['log', 1, 0.001, 1000],
        zoneentire_sy_southcent=['log', 1, 0.001, 1000],
        zoneentire_sy_vom=['log', 1, 0.001, 1000],
        zoneentire_sy_westside=['log', 1, 0.001, 1000],
        zoneentire_vk_aguacal=['log', 1, 0.001, 1000],
        zoneentire_vk_bay=['log', 1, 0.001, 1000],
        zoneentire_vk_cityson=['log', 1, 0.001, 1000],
        zoneentire_vk_eastside=['log', 1, 0.001, 1000],
        zoneentire_vk_highlands=['log', 1, 0.001, 1000],
        zoneentire_vk_kenwood=['log', 1, 0.001, 1000],
        zoneentire_vk_southcent=['log', 1, 0.001, 1000],
        zoneentire_vk_vom=['log', 1, 0.001, 1000],
        zoneentire_vk_westside=['log', 1, 0.001, 1000],

    )

    print('Setting Parvals for those listed in set_zonemult_start_values')
    for k in parvals.keys():
        c = df.loc[:, 'parnme'] == k
        assert c.sum() == 1, f"There is no matchinv parnme for {k}"
        df.loc[c, ['partrans', 'parval1', 'parlbnd', 'parubnd']] = parvals[k]
        df.loc[c, ['pargp', ]] = '_'.join(k.split('_')[0:2])

    print(
        df.loc[df.parnme.isin(list(parvals.keys())), ['parnme', 'partrans', 'parval1', 'parlbnd', 'parubnd', 'pargp']])

    return df


def set_zonemult_start_values(df):
    parvals = dict(
        zonemult_hk1_eastside=10.0,
        zonemult_hk2_eastside=1.0,
        zonemult_hk3_eastside=1.0,
        zonemult_hk4_eastside=1.0,
        zonemult_hk5_eastside=1.0,
        zonemult_hk6_eastside=1.0,
        zonemult_hk1_cityson=1.0,
        zonemult_hk2_cityson=1.0,
        zonemult_hk3_cityson=1.0,
        zonemult_hk4_cityson=1.0,
        zonemult_hk5_cityson=1.0,
        zonemult_hk6_cityson=1.0,
        zonemult_hk1_southcent=1.0,
        zonemult_hk2_southcent=0.5,
        zonemult_hk3_southcent=0.5,
        zonemult_hk4_southcent=0.5,
        zonemult_hk5_southcent=0.5,
        zonemult_hk6_southcent=0.5,
        zonemult_hk1_westside=1.0,
        zonemult_hk2_westside=0.5,
        zonemult_hk3_westside=0.5,
        zonemult_hk4_westside=0.5,
        zonemult_hk5_westside=0.5,
        zonemult_hk6_westside=0.5,
    )

    print('Setting Parvals for those listed in set_zonemult_start_values')
    for k in parvals.keys():
        c = df.loc[:, 'parnme'] == k
        assert c.sum() == 1, f"There is no matchinv parnme for {k}"
        df.loc[c, ['parval1']] = parvals[k]

    print(df.loc[df.parnme.isin(list(parvals.keys())), ['parnme', 'partrans', 'parval1', 'parlbnd', 'parubnd']])

    return df


def set_laymult_start_values(df):
    '''set partrans and parval1 in the parameter data df'''

    assert isinstance(df, pd.DataFrame)

    # ['partrans', 'parval1', 'parlbnd', 'parubnd']
    parvals = dict(
        laymult_drn_k=['log', (500 * 500) * .0002, (500 * 500) * .00001, (500 * 500) * .001],
        laymult_fmp_vk=['log', 0.0008873, 0.0001, .01, ],
        laymult_hk1=['log', 5, 1e-5, 1000, ],
        laymult_hk2=['log', 0.005, 1e-5, 100, ],
        laymult_hk3=['log', 0.005, 1e-5, 100, ],
        laymult_hk4=['log', .1, 1e-5, 100, ],
        laymult_hk5=['log', 0.01, 1e-5, 100, ],
        laymult_hk6=['log', 0.01, 1e-5, 100, ],
        laymult_ss1=['log', .05 / 200, 1e-7, .3 / 50, ],
        laymult_ss2=['log', 5e-5, 1e-6, 1e-3, ],
        laymult_ss3=['log', 5e-5, 1e-6, 1e-3, ],
        laymult_ss4=['log', 5e-6, 1e-6, 1e-3, ],
        laymult_ss5=['log', 5e-6, 1e-6, 1e-3, ],
        laymult_ss6=['log', 5e-6, 1e-6, 1e-3, ],
        laymult_sy1=['fixed', 0.3, .005, 1, ],
        laymult_vk1=['log', 0.1, 1e-3, 1e-1, ],
        laymult_vk2=['log', 1e-2, 1e-3, 1e-1, ],
        laymult_vk3=['log', 1e-2, 1e-3, 1e-1, ],
        laymult_vk4=['log', 1e-2, 1e-3, 1e-1, ],
        laymult_vk5=['log', 1e-2, 1e-3, 1e-1, ],
        laymult_vk6=['log', 1e-2, 1e-3, 1e-1, ],

        # laymult_hk=[1e-5, 1000, 1.],
        # laymult_vk=[1e-3, 1e-1, 1.],
        # laymult_ss=[1e-6, 1e-3, 1.],
        # laymult_sy=[0.0001, 0.3, 1.],
        # laymult_drn_k=[10, 50000, 5000],
        # laymult_fmp_vk=[0.0001, .01, 0.001],
    )

    print('Setting Parvals for those listed in set_laymult_start_values')
    for k in parvals.keys():
        c = df.loc[:, 'parnme'] == k
        assert c.sum() == 1, f"There is no matchinv parnme for {k}"
        df.loc[c, ['partrans', 'parval1', 'parlbnd', 'parubnd']] = parvals[k]

    print(df.loc[df.parnme.isin(list(parvals.keys())), ['parnme', 'partrans', 'parval1', 'parlbnd', 'parubnd']])

    return df


def get_zone_bounds():
    '''
    multiplier bounds for pilot points and zones
    '''
    zone_bounds = dict(sy1=[0.1, 100, 1.0],  # lo, hi, parval1
                       ss1=[0.001, 100, 1.0],
                       ss2=[0.001, 100, 1.0],
                       ss3=[0.001, 100, 1.0],
                       ss4=[0.001, 100, 1.0],
                       ss5=[0.001, 100, 1.0],
                       ss6=[0.001, 100, 1.0],
                       vk1=[0.001, 100, 1.0],
                       vk2=[0.001, 100, 1.0],
                       vk3=[0.001, 100, 1.0],
                       vk4=[0.001, 100, 1.0],
                       vk5=[0.001, 100, 1.0],
                       vk6=[0.001, 100, 1.0],
                       hk1=[0.001, 100, 1.0],
                       hk2=[0.001, 100, 1.0],
                       hk3=[0.001, 100, 1.0],
                       hk4=[0.001, 100, 1.0],
                       hk5=[0.001, 100, 1.0],
                       hk6=[0.001, 100, 1.0],
                       drn_k=[0.001, 100, 1.0],
                       fmp_vk=[0.0001, 100, 1.0], )

    return zone_bounds


def get_parbounds():
    '''parameter bounds for OTHER parameters'''
    parbounds = dict(sfr=[1e-9, 1000, 1e-2],  # lo, hi, initial values
                     fieswp=[0.5, 0.999, 0.7],

                     hfb=[0.000001, 1000000, 1000],
                     fmp_kc=[1 / 1.3, 1.3, 1.0],  # individual kc multipliers
                     fmp_ofe=[0.5, 1.0, 0.7, ],  # individual OFE values. vineyards are set in PEST.ipynb at 0.95
                     fmp_sfac=[1 / 1.3, 1.3, 1.0],  # multiplier for crop kc
                     rurfac=[.8, 1.25, 1.0],
                     ghbk=[0.0001, 10000, 1.4E-02],

                     )  # all sfac/kc multipliers

    return parbounds


def get_bounds():
    '''actual paremter bounds to be enforced at time of writing arrays'''
    bounds = dict(sy1=[0.0000001, 0.3],  # lo, hi
                  ss1=[1e-7, 0.3 / 50],
                  ss2=[1e-7, 1e-2],
                  ss3=[1e-7, 1e-2],
                  ss4=[1e-7, 1e-2],
                  ss5=[1e-7, 1e-2],
                  ss6=[1e-7, 1e-2],
                  vk1=[1e-5, 1],
                  vk2=[1e-5, 1],
                  vk3=[1e-5, 1],
                  vk4=[1e-5, 1],
                  vk5=[1e-5, 1],
                  vk6=[1e-5, 1],
                  hk1=[1.1e-10, 1e3],
                  hk2=[1.1e-10, 1e3],
                  hk3=[1.1e-10, 1e3],
                  hk4=[1.1e-10, 1e3],
                  hk5=[1.1e-10, 1e3],
                  hk6=[1.1e-10, 1e3],
                  drn_k=[(500 * 500) * .00001, (500 * 500) * .01],
                  fmp_vk=[0.0001, .01],
                  )
    return bounds


def set_crop_depth_irr_obs(df):
    # Sample data setup
    irr_depth = {'BareLand': 0.0,
                 'CitrusSubtropic': 1.9,  # some years are zero Q...
                 'DeciduousFruits': 1.5,
                 'FieldCrop': 2.0,
                 'GrainHayCrops': 2.0,
                 'Idle': 0.0,
                 'NativeVegetation': 0.0,
                 'Pasture': 4.0,
                 'SemiAgricultural': 0.0,
                 'SemiPaved': 0.0,
                 'TruckNursery': 0.6,
                 'Turf': 3.,
                 'Vineyard': 0.6,
                 'Walnuts': 2.0,
                 'Water': 0.0}

    def update_obsval_weight(row, irr_depth):
        for key in irr_depth:
            if 'depth' in row['obsnme'] and key.lower() in row['obsnme']:
                row['obsval'] = irr_depth[key]
                row['weight'] = 1.0 * row['weight']
                row['standard_deviation'] = 0.5

                if 'vine' in key.lower():
                    row['weight'] = 10. * row['weight']
                    row['standard_deviation'] = 0.1

                if int(row['obsnme'].split('date:')[1][0:4]) <= 1975:
                    row['weight'] = 0

        return row

    # Apply the function to each row
    df = df.apply(update_obsval_weight, axis=1, irr_depth=irr_depth)

    return df


def get_prefix_dict_for_pilot_points():
    prefix_dict = {0: ["hk1", 'ss1', "sy1", "vk1", 'fmp_vk', 'drn_k'],
                   1: ["hk2", "ss2", "vk2"],
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

    # todo add drain locations for model arrays
    g = os.path.join(folder, 'model_arrays', 'drain_locs.csv')
    drain_off = np.genfromtxt(g, delimiter=',', dtype=int)
    csv_data.loc[:, 'drain_off'] = 0
    csv_data.loc[:, 'drain_off'] = drain_off[csv_data.loc[:, 'i'] - 1, csv_data.loc[:, 'j'] - 1]
    c = csv_data.loc[:, 'drain_off'] == 0
    csv_data.loc[c, 'k'] = 0.0

    print(f"number of cells with >0 cond: {csv_data.query('k>0').shape[0]}")

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

    #set all drains to layer 1
    csv_data.loc[:, 'lay'] = 1

    with open(os.path.join(folder, 'drt.drt'), 'w') as w:
        w.write(head)
        w.write(head2)
        csv_data.to_csv(w, lineterminator='\n', index=False, header=False, sep='\t')
        for i in range(800):
            w.write('-1\n')


def write_kc(folder):
    ##read multiplier dict as
    multiplier_dict = pd.read_csv(os.path.join(folder, 'kc_scale_factors.csv'), index_col=0)
    multiplier_dict.index = multiplier_dict.index.str.strip()
    multiplier_dict = multiplier_dict.loc[:, 'parvalue'].to_dict()
    kcfile_in = os.path.join(folder, 'fmp_input', 'model_arrays', 'kc_GSP.txt')
    kcfile_out = os.path.join(folder, 'fmp_input', 'model_arrays', 'kc_GSP_PEST.txt')
    SFAC = multiplier_dict['SFAC']
    # Open the input file and prepare the output file
    with open(kcfile_in, 'r') as infile, open(kcfile_out, 'w') as outfile:
        for line in infile:
            parts = line.split()

            # Check if the line starts a new section
            if "SFAC" in line:  # write scale factor for a KC values
                # current_multiplier = multiplier_dict[parts[1]]
                line = f"{parts[0]}\t{SFAC}\t{parts[2]}\t{parts[3]}\n"
                outfile.write(line)  # Write the header line as-is
            else:  #multiply monthly kc value by kc scale factor eg kc_vineyards
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
    ofe = os.path.join(folder, 'ofe_scale_factors.csv')
    print(f'reading {ofe} for OFE factors')
    multiplier_dict = pd.read_csv(ofe, index_col=0)
    vals = multiplier_dict.parvalue.values

    nfarms = 139
    # Create a DataFrame by repeating the values and adjusting for the number of rows
    df = pd.DataFrame([vals] * nfarms, index=np.arange(1, nfarms + 1))

    outfile = os.path.join(folder, 'fmp_update', 'sv_OFE_GSP.txt')
    with open(outfile, 'w') as out:
        for date in pd.date_range('12/1/1969', freq='ME', end='1/1/2026'):
            out.write(f"SFAC  1.0   {date.strftime('%Y-%b')}\n")
            # for i in range(nfarms):
            #     out.write(vals)
            # df.to_csv(out, mode = 'append')
            out.write(df.to_string(index=True, header=False))
            out.write('\n')  # Optional: add a newline after the DataFrame

    print(f'done writing OFE to {outfile}')


def write_pilot_point(layer, prop, model_ws, skip_writing_output=False):
    if layer == 0:
        factors_file = os.path.join(model_ws, 'pp2024', "pp.fac")
    else:
        factors_file = os.path.join(model_ws, 'pp2024', "pp2.fac")

    # out_file = os.path.join(model_ws, 'pp2024_out', f"{prop}.txt")
    out_file = None

    pp_file = os.path.join(model_ws, 'pp2024', f"{prop}pp.dat")
    assert os.path.exists(pp_file), f"pp_file does not exist {pp_file}"
    print(f"pp_file = {pp_file}, factors_file={factors_file}, out_file={out_file}")

    # write the pilot points file here, but it is not actually used (just using the in-memory array file instead, below)
    hk_arr = pyemu.geostats.fac2real(pp_file, factors_file=factors_file, out_file=out_file)

    # load the zone multiplier and layer multiplier
    mul_f = os.path.join(model_ws, 'zone_pest_mult', f"zonemult_{prop}.csv")
    assert os.path.exists(mul_f), f"{mul_f} does not exist"
    mult = np.genfromtxt(mul_f, delimiter=',')

    lay_f = os.path.join(model_ws, 'zone_pest_mult', f"laymult_{prop}.csv")
    assert os.path.exists(lay_f), f"{lay_f} does not exist"
    lay = np.genfromtxt(lay_f, delimiter=',')

    assert mult.shape == lay.shape == hk_arr.shape, f"shapes are not equal\nmult.shape=={mult.shape}\nlay.shape=={lay.shape}\nhk_arr.shape=={hk_arr.shape}\n"

    lay_f = os.path.join(model_ws, 'zone_pest_mult', f"zoneentire_{prop}.csv")
    assert os.path.exists(lay_f), f"{lay_f} does not exist"
    zoneentire = np.genfromtxt(lay_f, delimiter=',')

    out = hk_arr * mult * lay * zoneentire

    # set parameter bounds
    bounds = get_bounds()[prop]
    out[out < bounds[0]] = bounds[0]
    out[out > bounds[1]] = bounds[1]
    # final values:
    array_out = os.path.join(model_ws, 'pp2024_out', f"{prop}.txt")

    if not skip_writing_output:
        np.savetxt(array_out, out)
        print(f"final array written to {array_out}")

    return out, lay, mult, hk_arr, zoneentire


def write_all_pp(model_ws, skip_writing_output=False):
    '''write outputs from pilots points, creating actual grid files'''

    prefix_dict = get_prefix_dict_for_pilot_points()

    for lay in prefix_dict.keys():
        for par in prefix_dict[lay]:
            write_pilot_point(lay, par, model_ws, skip_writing_output=skip_writing_output)


def HK_extract(workspace):
    '''extract K values from observed hk values from model'''

    infile = os.path.join(workspace, 'hk_estimates', "hk_estimates_for_pest.xlsx")
    outfile = os.path.join(workspace, 'hk_estimates', "hk_estimates_for_pest_simulated_K.csv")

    obs_vals = pd.read_excel(infile, index_col=0)

    hk = [np.genfromtxt(os.path.join(workspace, 'pp2024_out', f'hk{i}.txt')) for i in range(1, 7)]
    hk = np.stack(hk)
    print('extracting HK values from model\n')
    for ind, row in obs_vals.iterrows():
        laytop = int(row['laytop'])
        laybot = int(row['laybot'])

        avg = hk[slice(laytop, laybot + 1, 1), row['i'], row['j']]
        obs_vals.loc[ind, 'hk_pest'] = np.log10(hmean(avg))
        obs_vals.loc[ind, 'arithmethic_mean'] = hmean(avg)
        obs_vals.loc[ind, 'harmonic_mean'] = np.mean(avg)
        print(
            f"wellname:{row['well']}\n\tlaytop - {laybot}, laybot-{laybot}\n\tlayer values:--\n\t\t{avg}\n\tactual value\n\t\t{np.mean(avg):.3g}\n")

    obs_vals.to_csv(outfile)

    return obs_vals


def summarize_budget(folder):
    '''summarize the budget output terms for cleaner ingetion into pest processing'''
    b = os.path.join(folder, 'output', "Budget.txt")
    bud = pd.read_csv(b, sep='\s+')

    listx = ['PERCENT_ERROR']
    # Define aggregation functions for each column
    agg_funcs = {col: ('mean' if col in listx else 'sum') for col in bud.columns}

    # Perform the groupby and aggregation
    bud = bud.groupby(pd.to_datetime(bud.DATE_START).dt.year).agg(agg_funcs)

    cols_ = ['STORAGE_IN', 'STORAGE_OUT', "DRT_OUT",
             'RURWELLS_OUT', "MNIWELLS_OUT", 'GHB_IN', 'RCH_IN', 'SFR_IN', 'SFR_OUT',
             'MNW2_IN', 'MNW2_OUT', 'FMP_WELLS_OUT', 'FMP_FNR_IN', 'FMP_FNR_OUT',
             'IN_OUT', 'PERCENT_ERROR']

    bud = bud.reindex(columns=cols_).fillna(0)

    bud.index = pd.to_datetime(bud.index, format='%Y')
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
    q.loc[:, 'Station'] = station
    q.loc[:, 'Date'] = (pd.to_timedelta(q.Time.astype(float), unit="D") + pd.to_datetime('11/30/1969'))
    q = q.astype({'Date': 'datetime64[ns]'})

    q.to_csv(outfile)
    c = q.Flow < 0.001 * 60 * 60 * 24  # all flows below this threshold will be replaced to this value (USGS does not report below)
    q.loc[c, 'Flow'] = 0.001 * 60 * 60 * 24  # ft^3/day
    q.loc[:, "Flow"] = q.Flow.apply(np.log10)

    q.to_csv(outfile.replace(".csv", '_log.csv'))
    return q


def sfr_flow_accum(folder, station):
    '''load the observed flow to get the dates that the USGS gauge is active. must have the observed sfr file in the main dir'''
    outfile = os.path.join(folder, 'output', f'sfr_{station}_modeled_fdc_curve.csv')
    outfile_min = os.path.join(folder, 'output', f'sfr_{station}_modeled_water_year_min.csv')

    print(f'\nprinting the fdc sfr for {station} to:\n{outfile}')
    print(f'printing the water min flow for sfr for {station} to:\n{outfile_min}')

    obs_flow = pd.read_csv(os.path.join(folder, f'sfr_{station}_obs.csv'), index_col=0)
    obs_flow = obs_flow.astype({'date': 'datetime64[ns]', 'Water Year': 'datetime64[ns]'}).drop(columns='Q')

    print(f"loading sfr for {os.path.join(folder, f'sfr_{station}_obs.csv')}\n")

    sfr = os.path.join(folder, 'output', f"{station}_sfr_reformat.csv")
    flow = pd.read_csv(sfr, index_col=0)
    flow = flow.astype({'Date': 'datetime64[ns]'})

    # fix low values for modeled data
    c = flow.Flow < 0.001 * 60 * 60 * 24  # all flows below this threshold will be replaced to this value (USGS does not report below)
    flow.loc[c, 'Flow'] = 0.001 * 60 * 60 * 24  # ft^3/day

    flow = pd.merge(obs_flow, flow, left_on='date', right_on='Date')

    fdf = {x: np.percentile(flow.Flow.values, x) for x in range(5, 101, 5)}
    fdf = pd.DataFrame.from_dict(fdf, orient='index', columns=['term'])
    fdf.loc[:, 'term'] = fdf.term.apply(np.log10)
    fdf.loc[:, 'Station'] = station
    fdf.loc[:, 'desc'] = 'flow duration curve value'
    fdf.index.name = 'fdc_percentile'

    fdf.to_csv(outfile)
    print(fdf.head().loc[:, ['term', 'Station', 'desc']])

    wymin = flow.groupby('Water Year').min()
    wymin = wymin.drop(columns=['Date', 'date'])

    wymin.loc[:, 'term'] = wymin.Flow.apply(np.log10)
    wymin.loc[:, 'Date'] = wymin.index
    wymin.loc[:, 'Station'] = station
    wymin.loc[:, 'desc'] = 'yearly_min_value'
    wymin.index.name = 'yearly_min_value'
    wymin.to_csv(outfile_min)
    print(wymin.head().loc[:, ['Date', 'term', 'Station', 'desc']])

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
    out = flopy.utils.HydmodObs(os.path.join(workspace, 'output', 'SV_hyd.hyd'), )
    out = out.get_dataframe(start_datetime='12/1/1969')
    out = out.rename(columns=lambda x: x[6:] if len(x) > 12 else x)
    out = out.drop(columns='totim')

    out = out.reindex(pd.date_range('1975-01-01', freq='MS', periods=525))

    return out


def create_diff_for_hydobs(df, numper_diff=5):
    '''will take the drawdown of very (numper_diff) nth measurement (the zeroth and every numper_diff will be used as drawdown head)'''
    df = df.apply(calculate_differences, n=numper_diff)

    return df


def down_sample_hydobs(df, numper_diff=5, keep_every=4):
    "remove numper_diff measurements. then it will keep keep_every nth value"
    df = df.drop(df.iloc[::numper_diff, :].index)
    df = df.iloc[::keep_every, :]

    return df


def create_obs_from_hyd(sim):
    '''converting super large file of heads observations from wide to long to be loaded via pestpp'''
    sim = sim.stack()
    sim.index = sim.index.set_names(['date', 'Station', ])
    sim = sim.swaplevel().to_frame('meas').reset_index()

    return sim


def calculate_differences(series, n):
    '''function to create differences at every n for the hydmobs datasets'''
    differences = []
    for i in range(0, len(series), n):
        chunk = series[i:i + n]
        first_value = chunk[0]
        diff = [value - first_value for value in chunk]
        differences.extend(diff)
    return differences


def rolling_mean(df, nyears=None, nmonths=None):
    '''rolling mean of hydobs/and gwle data. dates are labeled as end of period. for 10-year it is past end of model, but just represents periods>2015-12-31'''

    if isinstance(nyears, int):
        window = 12 * nyears
    elif isinstance(nmonths, int):
        window = nmonths
    else:
        raise ValueError('nmonths and nyears are None')

    print(f"window: {window}")
    c = (df.rolling(window, min_periods=1, ).count() == window).all(axis=1)
    rn = df.rolling(window, min_periods=1, ).mean()  #remove rows without window number of measurements
    rn = rn.loc[c]

    #now revers the index, take window freq measurements, then reverse, and filter the final
    ind = rn.index
    ind = ind[::-1][::window][::-1]
    rn = rn.loc[ind]

    return rn


def limit_hyd_obs_date_range(df, end_of_model='2018-09-30'):
    df = df.loc[:end_of_model, :]
    return df


def run_all_hyd_obs(workspace):
    bigobj = load_hydobs(workspace)
    abs_small = down_sample_hydobs(bigobj, numper_diff=5, keep_every=4)
    abs_obs = create_obs_from_hyd(abs_small)

    print(f'there are {abs_obs.shape[0]} observations in the GWLE observation absolute elev file')

    diff_big = create_diff_for_hydobs(bigobj, numper_diff=5)
    diff_small = down_sample_hydobs(diff_big, numper_diff=5, keep_every=4)
    diff_obs = create_obs_from_hyd(diff_small)

    print(f'there are {diff_obs.shape[0]} observations in the GWLE observation drawdown elev file')

    # rolling observations
    roll = rolling_mean(bigobj, nyears=8, nmonths=None)
    roll_obs = create_obs_from_hyd(roll)

    #rolling 18month
    roll_short = rolling_mean(bigobj, nyears=None, nmonths=18)
    roll_short_obs = create_obs_from_hyd(roll_short)

    f_abs = os.path.join(os.path.join(workspace, "GWLE_OBS", 'gwle_asbolute_mod_heads.csv'))
    f_abs_dup = os.path.join(os.path.join(workspace, "GWLE_OBS", 'gwle_asbolute_mod_heads_duplicate.csv'))
    print(f"writing absolute heads to {f_abs}")
    f_diff = os.path.join(os.path.join(workspace, "GWLE_OBS", 'gwle_drawdown_mod_heads.csv'))
    print(f"writing drawdon heads to {f_diff}")
    f_roll = os.path.join(os.path.join(workspace, "GWLE_OBS", 'gwle_rolling_mod_heads.csv'))
    print(f"writing rolling heads to {f_roll}")
    f_roll_short = os.path.join(os.path.join(workspace, "GWLE_OBS", 'gwle_rolling_18month_mod_heads.csv'))
    print(f"writing rolling heads to {f_roll}")

    abs_obs.to_csv(f_abs)
    abs_obs.to_csv(f_abs_dup)
    diff_obs.to_csv(f_diff)
    roll_obs.to_csv(f_roll)
    roll_short_obs.to_csv(f_roll_short)

    print("Done writing to files")


def get_zone_bud(workspace):
    '''
    process zone budget of sv modflow output
    '''

    start_datetime_df = '12/1/1969'
    print(f'the start date time for the zone budget dataframe is {start_datetime_df}')

    zones_2020 = np.ones([6, 275, 85], dtype=int)

    #todo fix zonation_gwbasin_lay_

    for lay in np.arange(0, 6):
        zones_2020[lay, :, :] = np.genfromtxt(os.path.join(workspace, 'zbud', f'zonation_gwbasin_lay_{lay + 1}.csv'),
                                              delimiter=',')

    aliases = {1: 'Exterior', 2: 'Basin'}

    cb_f = os.path.join(workspace, 'output', 'sv_model_grid_6layers.cbb')
    cbb = bf.CellBudgetFile(cb_f, verbose=False)

    # allones = np.ones((ml.dis.nrow,ml.dis.ncol),dtype = int)
    zb_whole = flopy.utils.ZoneBudget(cbb, zones_2020, aliases=aliases)
    zb_df = zb_whole.get_dataframes(start_datetime=start_datetime_df, timeunit='D').multiply(1 / 43560.)

    wy = water_year(zb_df.index.get_level_values(0))
    zb_df.loc[:, 'Water Year'] = pd.to_datetime(wy, format='%Y')
    zb_df = zb_df.set_index('Water Year', append=True)
    days = zb_df.index.get_level_values(0).daysinmonth
    zb_df.loc[:, 'Days'] = days
    zb_df = zb_df.multiply(zb_df.Days, axis='index')
    zb_df.index = zb_df.index.droplevel(0)
    zb_df = zb_df.groupby(['Water Year', 'name']).sum().loc[:, 'Basin'].unstack()

    zb_df = zb_df.unstack().to_frame('term')

    out_file = os.path.join(workspace, 'output', 'zbud.csv')
    zb_df.to_csv(out_file)


def crop_irr_depth(workspace):
    '''
    calculate crop irr depth
    '''

    infile = os.path.join(workspace, 'output', "ByCrop.txt")
    outfile = os.path.join(workspace, 'output', "ByCrop_IRR_DEPTH_PEST.txt")
    fb = pd.read_csv(infile, sep='\s+')

    fb.loc[:, ['CU_INI', 'CU', 'CIR_INI', 'CIR', 'DEMAND_INI', 'DEMAND',
               'ADDED_DEMAND_INI', 'ADDED_DEMAND', 'TOT_DEEP_PERC', 'TOT_SURF_RUNOFF',
               'ADDED_DMD_DPERC', 'ADDED_DMD_RUNOFF', 'TRAN_POT', 'ANOXIA_LOSS',
               'SOIL_STRESS_LOSS', 'TRAN', 'TRAN_SURF_INI', 'TRAN_SURF', 'TRAN_IRR',
               'TRAN_PRECIP', 'TRAN_GW', 'EVAP_IRR', 'EVAP_PRECIP', 'EVAP_GW', ]] = fb.loc[:,
                                                                                    ['CU_INI', 'CU', 'CIR_INI', 'CIR',
                                                                                     'DEMAND_INI', 'DEMAND',
                                                                                     'ADDED_DEMAND_INI', 'ADDED_DEMAND',
                                                                                     'TOT_DEEP_PERC', 'TOT_SURF_RUNOFF',
                                                                                     'ADDED_DMD_DPERC',
                                                                                     'ADDED_DMD_RUNOFF', 'TRAN_POT',
                                                                                     'ANOXIA_LOSS',
                                                                                     'SOIL_STRESS_LOSS', 'TRAN',
                                                                                     'TRAN_SURF_INI', 'TRAN_SURF',
                                                                                     'TRAN_IRR',
                                                                                     'TRAN_PRECIP', 'TRAN_GW',
                                                                                     'EVAP_IRR', 'EVAP_PRECIP',
                                                                                     'EVAP_GW', ]].mul(
        fb.loc[:, 'DELT'], axis=0).div(fb.loc[:, 'IRRIGATED_AREA'], axis=0)

    fb.loc[:, 'DATE_START'] = pd.to_datetime(fb.loc[:, 'DATE_START'])
    fb = fb.groupby("DATE_START   	CROP_NAME".split()).sum().loc[:, ['DEMAND']].unstack(1)
    fb.index = water_year(fb.index)
    fb = fb.groupby(level=0).sum()
    fb.index = pd.to_datetime(fb.index, format="%Y")
    fb = fb.droplevel(0, 1)
    fb = fb.rename(columns=land_use_renamed)
    fb.index.name = 'date'
    fb.columns.name = 'CROP'
    fb.unstack().to_frame('term').to_csv(outfile)

    return fb


def total_irr_demand(workspace):
    '''calculate total volumetric demand per year'''
    infile = os.path.join(workspace, 'output', "ByCrop.txt")
    outfile = os.path.join(workspace, 'output', "ByCrop_TOTAL_DEMAND_PEST.txt")
    fb = pd.read_csv(infile, sep='\s+')

    fb.loc[:, ['CU_INI', 'CU', 'CIR_INI', 'CIR', 'DEMAND_INI', 'DEMAND',
               'ADDED_DEMAND_INI', 'ADDED_DEMAND', 'TOT_DEEP_PERC', 'TOT_SURF_RUNOFF',
               'ADDED_DMD_DPERC', 'ADDED_DMD_RUNOFF', 'TRAN_POT', 'ANOXIA_LOSS',
               'SOIL_STRESS_LOSS', 'TRAN', 'TRAN_SURF_INI', 'TRAN_SURF', 'TRAN_IRR',
               'TRAN_PRECIP', 'TRAN_GW', 'EVAP_IRR', 'EVAP_PRECIP', 'EVAP_GW', ]] = fb.loc[:,
                                                                                    ['CU_INI', 'CU', 'CIR_INI', 'CIR',
                                                                                     'DEMAND_INI', 'DEMAND',
                                                                                     'ADDED_DEMAND_INI', 'ADDED_DEMAND',
                                                                                     'TOT_DEEP_PERC', 'TOT_SURF_RUNOFF',
                                                                                     'ADDED_DMD_DPERC',
                                                                                     'ADDED_DMD_RUNOFF', 'TRAN_POT',
                                                                                     'ANOXIA_LOSS',
                                                                                     'SOIL_STRESS_LOSS', 'TRAN',
                                                                                     'TRAN_SURF_INI', 'TRAN_SURF',
                                                                                     'TRAN_IRR',
                                                                                     'TRAN_PRECIP', 'TRAN_GW',
                                                                                     'EVAP_IRR', 'EVAP_PRECIP',
                                                                                     'EVAP_GW', ]].mul(
        fb.loc[:, 'DELT'], axis=0) / 43560

    fb.loc[:, 'DATE_START'] = pd.to_datetime(fb.loc[:, 'DATE_START'])
    fb = fb.groupby("DATE_START   	CROP_NAME".split()).sum().loc[:, ['DEMAND']].unstack(1)
    fb.index = water_year(fb.index)
    fb = fb.groupby(level=0).sum()
    fb.index = pd.to_datetime(fb.index, format="%Y")
    fb = fb.droplevel(0, 1)
    fb.index.name = 'date'
    fb.columns.name = 'CROP'
    fb = fb.rename(columns=land_use_renamed)
    fb.unstack().to_frame('term').to_csv(outfile)

    return fb


land_use_renamed = {
    "BARE_LAND": "BareLand",
    "CITRUS_AND_SUBTROPIC": "CitrusSubtropic",
    "DECIDUOUS_FRUITS_AND": "DeciduousFruits",
    "FIELD_CROP": "FieldCrop",
    "GRAIN_AND_HAY_CROPS": "GrainHayCrops",
    "IDLE": "Idle",
    "NATIVE_VEGETATION_RI": "NativeVegetation",
    "PASTURE": "Pasture",
    "SEMIAGRICULTURAL": "SemiAgricultural",
    "SEMIPAVED": "SemiPaved",
    "TRUCK_NURSERY_AND_BE": "TruckNursery",
    "TURF": "Turf",
    "VINEYARD": "Vineyard",
    "WALNUTS": "Walnuts",
    "WATER": "Water"
}


def water_year(date):
    '''
	this returns an integer water year of the date
	'''

    def wy(date):
        if date.month < 10:
            return date.year
        else:
            return date.year + 1

    if isinstance(date, pd.Series):
        return date.apply(wy)
    if isinstance(date, datetime):
        return wy(date)
    elif isinstance(date, pd.DatetimeIndex):
        return [wy(i) for i in date]
    else:
        import warnings
        warnings.warn('not a Series/datetime/DatetimeIndex object')
        # print('not a Series/datetime/DatetimeIndex object')
        return np.nan


def re_run_model_for_init(folder, turn_off=False):
    '''
    set model files to off/on and limit nper for re-running the model
    :param folder:
    :param sv_nam_file:
    :param sv_bas6_file:
    :param turn_off:
    :return: None
    '''

    sv_nam_file = 'SVIGFM_GSP.nam'
    sv_bas6_file = 'sv_model_grid_6layers_GSP.dis'
    # Modify the sv.nam file
    nam = os.path.join(folder, sv_nam_file)
    bas = os.path.join(folder, sv_bas6_file)

    if turn_off:
        nper = 121
    else:
        nper = 586

    print(
        f"\nChanging the following files for a re-run. setting turn_off=={turn_off} and nper={nper}\n\t{sv_nam_file} and \n\t{sv_bas6_file}\n")

    with open(nam, 'r') as file:
        lines = file.readlines()

    # Add '#' in front of lines containing 'WEL'
    with open(nam, 'w') as file:
        for line in lines:
            if 'SON.wel' in line:
                if turn_off:
                    file.write(f"# WEL 12 SON.wel\n")
                    print('turned OFF wel file')
                else:
                    file.write(f"WEL 12 SON.wel\n")
                    print('turned ON wel file')
            else:
                file.write(line)

    # Modify the sv.bas6 file
    with open(bas, 'r') as file:
        bas6_lines = file.readlines()

    # Change the 586 value on the second line
    second_line_parts = bas6_lines[1].split()
    second_line_parts[3] = str(nper)  # Assuming 586 is the 4th value (index 3)
    bas6_lines[1] = '  '.join(second_line_parts) + '\n'

    with open(sv_bas6_file, 'w') as file:
        file.writelines(bas6_lines)

    print(f'set bas6 to {nper} stress periods\n\n')


def pre_run(folder):
    re_run_model_for_init(folder, turn_off=True)

    if pyemu.os_utils.platform.system() == 'Windows':
        print('running pre-run')
        pyemu.os_utils.run(r'mf-owhm.exe SVIGFM_GSP.nam')
    else:
        print('running pre-run')
        pyemu.os_utils.run(r'mf-owhm.nix SVIGFM_GSP.nam')

    filename = os.path.join(folder, 'output', "sv_model_grid_6layers.hds")
    hdsobj = bf.HeadFile(filename)
    times = hdsobj.get_times()
    hds = hdsobj.get_data(totim=times[-1])
    print(f"setting starting heads from {pd.to_datetime('1/1/1969') + pd.to_timedelta(times[-1], unit='D')}")

    for lay in range(0, 6):
        head = hds[lay]
        print('writing heads to ', os.path.join(folder, 'init_heads', f"init_heads_lay{lay + 1}.dat"))
        np.savetxt(os.path.join(folder, 'init_heads', f"init_heads_lay{lay + 1}.dat"), head, fmt="%.0f")

    re_run_model_for_init(folder, turn_off=False)


def post_process(folder):
    # todo update aws to accomodate memory needs of zone budget
    # todo add zbud.csv to PEST.ipynb
    # get_zone_bud(folder)

    print("starting post processing\n")
    total_irr_demand(folder)
    crop_irr_depth(folder)
    print("Done with irrigation depths\n")
    HK_extract(folder)
    print("Done with HK_extract\n")
    summarize_budget(folder)
    print("Done with summarize_budget\n")
    _ = sfr_flows_log_transform(os.path.join(folder, 'output', "kenwood_sfr.dat"),
                                os.path.join(folder, 'output', "kenwood_sfr_reformat.csv"),
                                station='kenwood')
    print("Done with sfr_flows_log_transform\n")

    _ = sfr_flows_log_transform(os.path.join(folder, 'output', "agua_caliente_sfr.dat"),
                                os.path.join(folder, 'output', "aguacal_sfr_reformat.csv"),
                                station='aguacal')

    run_all_hyd_obs(folder)
    print("Done with run_all_hyd_obs\n")
    _ = sfr_flow_accum(folder, 'kenwood')
    _ = sfr_flow_accum(folder, 'aguacal')
    print("Done with sfr_flow_accum\n")

    # sfr_processing.run(folder)


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
    print('doing pre-run twice\n')
    pre_run(foldr)
    pre_run(foldr)
    print('Done with pre-runs\n')

    if pyemu.os_utils.platform.system() == 'Windows':
        print('running with windows executable')
        pyemu.os_utils.run(r'mf-owhm.exe SVIGFM_GSP.nam')
    else:
        print('running with linux executable')
        pyemu.os_utils.run(r'mf-owhm.nix SVIGFM_GSP.nam')

    post_process(foldr)

    print('DONE WITH RUN\n' * 5)
