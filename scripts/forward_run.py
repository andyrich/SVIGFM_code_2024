import os
import numpy as np
import pandas as pd
import pyemu


def read_drain(folder):
    # Step 1: Open the text file and read lines
    f = os.path.join(folder,'drt.drt')
    with open(f, 'r') as file:
        lines = file.readlines()

    # Step 2: Extract the number n from the 4th line after splitting by spaces
    n = int(lines[8].split()[0])

    # Step 3: Read the next n rows from a CSV file using pandas
    # Assuming the CSV file is named 'data.csv' and is in the same directory
    csv_data = pd.read_csv(f, skiprows=9, nrows=n, sep ='\s+',  names = ['lay', 'i', 'j', 'top', 'k', 'd', 'farm', 'd2', 'd3'])

    g = os.path.join(folder,'model_arrays', 'uzf_vk.uzf'  )
    vk = np.genfromtxt(g,delimiter = ',')

    csv_data.loc[:,'k'] = vk[csv_data.loc[:,'i']-1,csv_data.loc[:,'j']-1]
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
    if layer !=1:
        factors_file = os.path.join(model_ws,'pp2024', "pp.fac")
    else:
        factors_file = os.path.join(model_ws,'pp2024', "pp2.fac")

    out_file = os.path.join(model_ws, 'pp2024_out',f"{prop}.txt")
    
    pp_file =  os.path.join(model_ws, 'pp2024',f"{prop}pp.dat")
    assert os.path.exists(pp_file), f"pp_file does not exist {pp_file}"
    print(f"pp_file = {pp_file}, factors_file={factors_file}, out_file={out_file}")
    
    hk_arr = pyemu.geostats.fac2real(pp_file, factors_file=factors_file, out_file=out_file)

def write_all_pp(model_ws):
    prefix_dict= {0:["hk1","sy1","vk1"],
                 1:["hk2","ss2","vk2",'fmp_vk', 'drn_k'],
                 2:["hk3","ss3","vk3"],
                 3:["hk4","ss4","vk4"],
                 4:["hk5","ss5","vk5"],
                 5:["hk6","ss6","vk6"]}
    for lay in prefix_dict.keys():
        for par in prefix_dict[lay]:
            write_pilot_point(lay, par, model_ws)



if __name__=='__main__':
    foldr = os.getcwd()
    print(os.getcwd())
    
    read_drain(foldr)
    write_all_pp(foldr)