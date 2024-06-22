import flopy
from flopy.discretization.structuredgrid import StructuredGrid
import numpy as np
import pyemu

from pyemu.pst.pst_utils import (
    SFMT,
    IFMT,
    FFMT,
    pst_config,
    parse_tpl_file,
    try_process_output_file,
)


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