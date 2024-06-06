import os
import conda_scripts.arich_functions as af
import conda_scripts.sv_budget.load_sv_model
import flopy
import geopandas as gpd
from conda_scripts import arich_functions, plot_help

import os
# sys.path.append('c:\conda_scripts')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# import SRPHM

import pandas as pd

from conda_scripts import plot_help as ph
# import plot_wet as pw
from conda_scripts import  owhm

def setup(folder, test_name, note = None):
    if not os.path.exists(folder):
        os.mkdir(folder)

    p = os.path.join(folder, test_name)
    if not os.path.exists(p):
        os.mkdir(p)

    if note is not None:
        with open(os.path.join(p, 'note.txt'), 'w') as n:
            n.write(note)

    return p


def bar_comp(folder_name, name_test, yearly_df, yearly_owhm, ):
    comp = pd.merge(yearly_df.mean().drop('farm_wells').rename({'mnw2': 'farm_wells'}).to_frame('V1'),

                    yearly_owhm.mean().rename({'wellsv1': 'wells'}).to_frame('OWHM2'), left_index=True,
                    right_index=True,
                    how='outer').drop(
        ['mnw2', 'gw_et', 'constant_head', 'ndays', 'total', 'percent_discrepancy', 'head_dep_bounds', 'in-out'])

    ax = comp.plot.barh(grid=True)

    ax.set_title('average fluxes for SVIGFM V1 and OWHM2 beta')

    comp.to_html(os.path.join(folder_name, name_test, "bar_comp.html"))
    comp.to_excel(os.path.join(folder_name, name_test, "bar_comp.xlsx"))
    plt.savefig(os.path.join(folder_name, name_test, "bar_comp.png"), bbox_inches = 'tight')


def plot_ts(folder, test_name, yearly, owhm_yeary):
    drops = ['mnw2', 'gw_et', 'constant_head', 'ndays', 'total', 'percent_discrepancy', 'head_dep_bounds', 'in-out']
    left = yearly.drop(columns='farm_wells').rename(columns={'mnw2': 'farm_wells'})
    left = left.drop(columns=[x for x in drops if x in left.columns])
    left = left.loc[:,left.columns[left.sum().abs()>0]]

    right = owhm_yeary.rename(columns={'wellsv1': 'wells'})
    right = right.drop(columns=[x for x in drops if x in right.columns])
    right = right.loc[:, right.columns[right.sum().abs() > 0]]

    # right = right.reindex(columns=left.columns)
    cols = list(set(left.columns.tolist()) | set(right.columns.tolist()))

    left = left.reindex(columns = cols)
    right = right.reindex(columns=cols)
    fig, ax = plt.subplots(nrows=right.shape[1], sharex=True, figsize=(10, 10))
    plt.subplots_adjust(hspace=0)
    ax = ax.ravel()
    for n, col in enumerate(cols):
        ax[n].plot(left.index, left.loc[:, col], label='SVIGFM V1')
        ax[n].plot(right.index, right.loc[:, col], label='OWHM BETA')
        ax[n].grid(True)
        ax[n].annotate(col, (0, 0), va='bottom', xycoords='axes fraction', rotation=-0, size=12)

        # if n == 0:
        #     ax[0].legend()

        if col =='storage':
            ax[n].plot(left.index, left.loc[:, col].cumsum(), label='Cumulative SVIGFM V1', c= 'blue',marker = 'o')
            ax[n].plot(right.index, right.loc[:, col].cumsum(), label='Cumulative OWHM BETA', c = 'orange', marker = '.')
            ax[n].legend(loc = 'upper left', bbox_to_anchor = (1,1))

    plt.savefig(os.path.join(folder, test_name, "ts_comp.png"), bbox_inches = 'tight')


def plot_compare_q(folder_out, test, yearly_df_, owhm2_dfi, workspace):
    '''plot pumping per crop and compare with preivous model'''
    dts__ = load_dts()

    farmCrop = pd.read_csv(os.path.join(workspace, "output", "ByFarm_ByCrop.txt"), sep='\s+')
    farmCrop = pd.merge(farmCrop, dts__, left_on='PER', right_on='kstp')
    farmCrop.loc[:, 'gw'] = farmCrop.filter(like='_IRR').sum(axis=1)
    farmCrop = farmCrop.loc[farmCrop.loc[:, 'IRRIGATED_AREA'] > 0, :]
    farmCrop.loc[:, 'gw'] = farmCrop.loc[:, 'gw'] * farmCrop.loc[:, 'DELT'] / 43560
    farmCrop.loc[:, 'Water Year'] = arich_functions.water_year(farmCrop.date)
    farmCrop.loc[:, 'Water Year'] = pd.to_datetime(farmCrop.loc[:, 'Water Year'], format="%Y")
    tf = farmCrop.drop(columns='date').groupby(['Water Year', 'CROP_NAME']).sum().loc[:, ['gw']]
    # ax = .plot.bar(stacked = True)
    fig, ax = plot_help.stackedbar_wdates((tf.loc[:, 'gw']).unstack(), colormap=mpl.cm.tab20, plot_wet_bars = False)
    ax.set_title('Total gw pumping')
    ax.grid(True)
    (tf.loc[:, 'gw']).unstack().head()
    # ax.plot(yearlyv2.index, -yearlyv2.farm_wells,c = 'r',lw = 5, label = 'Total Irrigation Ag Pumping CIRNOQ=ON')
    # ax.plot(yearlyv3.index, -yearlyv3.farm_wells,c = 'b',lw = 3, label = 'Total Irrigation Ag Pumping CIRNOQ=OFF')
    ax.plot(yearly_df_.index, -yearly_df_.mnw2, c='orange', lw=3, label='Total Ag Pumping Version 1.0')
    ax.plot(owhm2_dfi.index, -owhm2_dfi.farm_wells, c='blue', lw=3, label='Total Ag Pumping OWHM Beta')
    # ax.plot(yearly_qmax.index, -yearly_qmax.mnw2,c = 'brown',lw = 3, label = 'Total Ag Pumping Version 1.0 (no qmax)')

    ax.set_ylim([0, 8000])

    ax.legend(bbox_to_anchor=(1, 1), loc='upper left',reverse = True)

    plt.savefig(os.path.join(folder_out, test, "q.png"), bbox_inches = 'tight')

def plot_converge_issues(folder, owhm_folder, test_name, ):

    hclose = pd.read_csv(os.path.join(owhm_folder, 'output', 'Conv_HCLOSE.txt'), sep='\s+')
    rclose = pd.read_csv(os.path.join(owhm_folder, 'output', 'Conv_RCLOSE.txt'), sep='\s+')

    hclose.loc[:,'i'] = hclose.loc[:,'ROW']-1
    hclose.loc[:, 'j'] = hclose.loc[:, 'COL']-1
    rclose.loc[:,'i'] = rclose.loc[:,'ROW']-1
    rclose.loc[:, 'j'] = rclose.loc[:, 'COL']-1

    _,mg, modgeom = conda_scripts.arich_functions.get_flopy_model_spatial_reference('son',return_shp = True)
    hclose = hclose.groupby(['i', 'j']).sum().loc[:, ['CHNG_HEAD']].reset_index()
    hclose = pd.merge(hclose, modgeom, on = ['i','j'])

    hclose = gpd.GeoDataFrame(hclose, geometry='geometry', crs = 2226)
    print(rclose.shape)
    rclose = rclose.groupby(['i', 'j']).sum().loc[:, ['FLOW_RESIDUAL']].reset_index()
    rclose = pd.merge(rclose, modgeom, on = ['i','j'])
    rclose = gpd.GeoDataFrame(rclose, geometry='geometry', crs = 2226)
    print(rclose.shape)


    fig = plt.figure( figsize  = (8.5,11))
    mm = conda_scripts.make_map.make_map('RCLOSE (flow-residual) issues')
    ax = mm.plotloc(fig, locname='SON_MOD')
    hclose.set_geometry(hclose.geometry.centroid).plot('CHNG_HEAD', ax = ax,legend = True, s = 50)
    plt.savefig(os.path.join(folder, test_name, "converge_hclose.png"), bbox_inches='tight')

    fig = plt.figure( figsize  = (8.5,11))
    mm = conda_scripts.make_map.make_map('rclose (head-residual) issues')
    ax = mm.plotloc(fig, locname='SON_MOD')
    rclose.set_geometry(rclose.geometry.centroid).plot('FLOW_RESIDUAL', ax = ax,legend = True, s = 10)
    plt.savefig(os.path.join(folder, test_name, "converge_rclose.png"), bbox_inches='tight')



def plot_stacked_bar(folder, test_name, yearly, owhm):
    ''' plot total water budget for both models'''
    d = ['constant_head', 'in-out', 'storage', 'total', 'percent_discrepancy']
    fig, ax = ph.stackedbar_wdates(yearly.drop(columns=d))
    # print(ax.get_xlim())
    ax.plot(yearly.index, -yearly.storage.cumsum(), c='k')
    ax.set_title("SVIGFM")
    plt.savefig(os.path.join(folder, test_name, "stacked_comp_main.png"), bbox_inches = 'tight')

    fig, ax = ph.stackedbar_wdates(owhm.drop(columns=d))
    ax.set_title("OHWM2 Beta")
    ax.plot(owhm.index, -owhm.storage.cumsum(), c='k')
    m = -owhm.storage.cumsum().min()
    ax.set_ylim([min(m,ax.get_ylim()[0]), None ])
    plt.savefig(os.path.join(folder, test_name, "stacked_comp_owhm.png"), bbox_inches = 'tight')

def plot_error(folder, test_name, error_base_inc, error_base_cum, error_owhm_inc,error_owhm_cum):
    '''plot instantaneous and cumulative water budget errors'''
    fig, (ax, ax3) = plt.subplots(2, 1, sharex=True)
    error_base_inc.percent_discrepancy.plot(label='incremental discrepancy', ax=ax)
    ax.legend(loc='lower left')

    ax2 = ax.twinx()
    error_base_cum.percent_discrepancy.plot(ax=ax2, c='g', label='cumulative discrepancy')
    ax2.legend(loc='lower right')
    ax.set_title('errors SVIGFM V1')

    error_owhm_inc.percent_discrepancy.plot(label='incremental discrepancy', ax=ax3)
    ax3.legend(loc='lower left')
    ax3.set_title('errors OWHM Beta')
    ax4 = ax3.twinx()
    error_owhm_cum.percent_discrepancy.plot(ax=ax4, c='g', label='cumulative discrepancy')
    ax4.legend(loc='lower right')

    plt.savefig(os.path.join(folder, test_name, "err.png"), bbox_inches='tight')



def plot_head_map(workspace, testname, base):
    hds = flopy.utils.binaryfile.HeadFile(os.path.join(workspace, 'output', 'sv_model_grid_6layers.hds'))

    hd = hds.get_alldata()
    # plt.figure(figsize = (10,30),dpi = 250)
    fig, ax = plt.subplots(2, 3, figsize=(10, 30), dpi=250)
    ax = ax.ravel()

    hd = np.ma.array(hd, mask=hd == -999.)

    for n in np.arange(hd[-1].shape[0]):
        ax[n].set_axis_off()
        ax[n].contourf(hd[-1, n], vmax=1000, vmin=-200, levels=np.arange(-200, 1000, 50), cmap='jet', origin="upper")


    plt.savefig(os.path.join(folder, test_name, f"heads_{testname}.png"), bbox_inches='tight')


def plot_hydros(workspace, testname, base, folder):
    ''' plot a grid of hydrographs, makes same figure for old and new model. '''
    ml = conda_scripts.sv_budget.load_sv_model.get_model(workspace=workspace)
    hds = flopy.utils.binaryfile.HeadFile(os.path.join(workspace, 'output', 'sv_model_grid_6layers.hds'))
    ii, jj = np.meshgrid(np.arange(ml.dis.ncol), np.arange(ml.nrow))

    GIS_out = r'C:\GSP\sv'
    gdb = os.path.join(GIS_out,"GIS", 'sv_model_geodatabase.gdb')
    active_grid = gpd.read_file(gdb, layer='active_grid_cells')
    active_grid_dis = gpd.read_file(gdb, layer='active_grid_cells_dissolve')
    active_grid_dis.insert(0, 'Sub', 'Sub')
    active_grid_dis = active_grid_dis.dissolve("Sub")
    subcat = gpd.read_file(gdb, layer='SCWA_sonoma_valley_BCM_subcatchments')
    sr, mg, modgeoms = conda_scripts.arich_functions.get_flopy_model_spatial_reference('sv', True)

    ibnd = ml.bas6.ibound.array[0]

    active_sub = np.genfromtxt(os.path.join(workspace,"fmp_input", 'farms', 'farm2016.dat'),
                               dtype= int)
    ii = np.where(ii % 15 == 0, ii, np.nan)
    jj = np.where(jj % 15 == 0, jj, np.nan)

    ii[~((ibnd > 0) | (active_sub > 0))] = np.nan
    jj[~((ibnd > 0) | (active_sub > 0))] = np.nan

    loc = np.array(list(zip(ii.ravel(), jj.ravel())))

    c = np.isnan(loc).any(axis=1)

    loc = loc[~c, :]
    if not os.path.exists(os.path.join(folder, testname,'hydros')):
        os.mkdir(os.path.join(folder, testname,'hydros'))

    fold = os.path.join(folder, testname,'hydros')

    for c, v in enumerate(loc):
        print(f"plotting {c} of {len(loc)}", end='\r')
        fig, (ax, ax1) = plt.subplots(2, 1)
        active_grid_dis.exterior.plot(ax=ax1)
        subcat.plot(ax=ax1, fc='None', ec='g')
        modgeoms.set_geometry(modgeoms.geometry.centroid).query(f"i=={v[1]} and j=={v[0]}").plot(ax=ax1, fc='k')

        for lay in range(6):
            # n =ml.dis.get_node([lay,int(v[0]), int(v[1])])

            trans = hds.get_ts([[lay, int(v[1]), int(v[0])]])
            trans[trans[:, 1] == -999, :] = np.nan
            if not np.isnan(trans[:, 1]).all():
                days = pd.to_datetime('12/1/1969') + pd.to_timedelta(trans[:, 0], unit="D")

                ax.plot(days, trans[:, 1], label=lay + 1, ls='-')

        ax.legend(title='Layer', bbox_to_anchor=(1, 1), loc='upper left');
        ax.grid(True)
        ax1.set_axis_off()
        ax.set_facecolor('lightgrey')

        if base:
            base_str = 'base'
        else:
            base_str = 'ver'
        plt.savefig(os.path.join(fold, f'hydro_{v[0]}_{v[1]}_{base_str}.png'))


def plot_fb_details(folder, test_name, fb, raw, base):
    '''plot farm budget details'''

    if base:
        base_str = 'base'
    else:
        base_str = 'ver'

    for cols in np.arange(0, raw.shape[1], 5):
        raw.query("FID>80").groupby(level=[0, 1]).sum().droplevel(1, 0).iloc[:, cols:cols + 5].plot(subplots=True)
        plt.savefig(os.path.join(folder, test_name, f"fbdetails_{cols}_{base_str}.png"), bbox_inches='tight')


    #plot farms
    q = raw.copy()
    q = q.unstack().loc[:, q.unstack().columns[q.unstack().sum().abs() > .05]].stack()
    q = q.query('FID<72').groupby(level=[0]).sum().drop(columns=['Q-tot-in', 'Q-tot-out', 'dlength'])
    for c in q.columns:
        if 'out' in c:
            q.loc[:, c] = q.loc[:, c] * -1

    q = q.filter(regex='Q')
    q = q.groupby(af.water_year(q.index)).sum()
    q.index = pd.to_datetime(q.index, format="%Y")
    fig, ax = plot_help.stackedbar_wdates(q, colormap='tab20')
    ax.set_title('farm budget for active farms')
    plt.savefig(os.path.join(folder, test_name, f"farmbud_{base_str}.png"), bbox_inches='tight')

    q.to_excel(os.path.join(folder, test_name, f"farm budget for active farms {base_str}.xlsx"))

    if not base:
        ## plot mtn fronts
        q = raw.copy()
        q = q.unstack().loc[:, q.unstack().columns[q.unstack().sum().abs() > .05]].stack()
        q = q.query('FID>80').groupby(level=[0]).sum().drop(columns=['Q-tot-in', 'Q-tot-out', 'dlength'])

        for c in q.columns:
            if 'out' in c:
                q.loc[:, c] = q.loc[:, c] * -1

        q = q.groupby(af.water_year(q.index)).sum()
        q.index = pd.to_datetime(q.index, format="%Y")

        fig, ax = plot_help.stackedbar_wdates(q, )
        ax.set_title('farm budget for subcatchment farms')

        plt.savefig(os.path.join(folder, test_name, f"farmbud_subcat_{base_str}.png"), bbox_inches='tight')
        q.to_excel(os.path.join(folder, test_name, f"farm budget for subcatchment farms {base_str}.xlsx"))



def __f(df):
    '''helper function'''
    q = df.copy()
    q = q.unstack().loc[:, q.unstack().columns[q.unstack().sum().abs() > .05]].stack()
    q = q.query('FID<80').groupby(level=[0]).sum().drop(columns=['Q-tot-in', 'Q-tot-out', 'dlength'])

    for c in q.columns:
        if 'out' in c:
            q.loc[:, c] = q.loc[:, c] * -1

    q = q.groupby(af.water_year(q.index)).sum()
    q.index = pd.to_datetime(q.index, format="%Y")

    return q


def plot_by_crop(folder, workspace, testname):
    '''plot numerous plots from ByCrop. one for each output vaiable.'''
    path = os.path.join(workspace, 'output', "ByCrop.txt")

    byc = pd.read_csv(path, sep='\s+')
    byc = byc.groupby('CROP_NAME').sum().drop(columns=['PER', 'STP', 'CROP', 'DELT', 'DYEAR', 'DATE_START'])

    if not os.path.exists(os.path.join(folder, testname,'crop_water_use')):
        os.mkdir(os.path.join(folder, testname,'crop_water_use'))

    for name, row in byc.T.iterrows():
        plt.figure()
        row.plot.barh(title=name)
        plt.savefig(os.path.join(folder, testname,'crop_water_use', name+'.png'), bbox_inches = 'tight')

    byc.to_excel(os.path.join(folder, testname,'crop_water_use', 'crop_water_use.xlsx'))


def comp_fb_details(folder, test_name, raw_base, raw_owhm):
    '''compare individual terms in fb details'''
    q_base = __f(raw_base)
    q_ver = __f(raw_owhm)

    b = q_base.columns.tolist()
    v = q_ver.columns.tolist()

    cols = list(set(b) & set(v))

    fig, ax = plt.subplots(nrows=int(len(cols)/2)+1, ncols=2, sharex=True, figsize=(10, 10))
    plt.subplots_adjust(hspace=0)
    ax = ax.ravel()
    for n, col in enumerate(cols):
        ax[n].plot(q_base.index, q_base.loc[:, col], label='SVIGFM V1')
        ax[n].plot(q_ver.index, q_ver.loc[:, col], label='OWHM BETA')
        ax[n].grid(True)
        ax[n].annotate(col, (0, 0), va='bottom', xycoords='axes fraction', rotation=-0, size=12)

        if n == len(cols)-1:
            ax[n].legend(loc = 'upper left', bbox_to_anchor = (1,1))

    plt.savefig(os.path.join(folder, test_name, "fb_comp.png"), bbox_inches = 'tight')

def load_dts():
    dts = pd.DataFrame([pd.date_range("12-01-1969", freq="ME", periods=587), ], index=['date']).T
    dts.loc[:, 'ndays'] = dts.date.dt.days_in_month
    dts.loc[:, 'kstp'] = np.arange(dts.shape[0]) + 1

    return dts


def load_bud(workspace, folder, test_name, base = True, hard_load = False):

    if base:
        name = 'base'
    else:
        name = 'owhm'

    if base:
        p = os.path.join('base_model_data',  f'{name}_yearly.pickle')
        loc =os.path.join('base_model_data',)
    else:
        loc = os.path.join(folder, test_name,)
        p = os.path.join(loc, f'{name}_yearly.pickle')


    if  os.path.exists(p) and not hard_load:

        print(f'loading data from {loc}')
        cum_wy = pd.read_pickle(os.path.join(loc, f'{name}_cum_wy.pickle'))
        incremental_ = pd.read_pickle(os.path.join(loc, f'{name}_incremental.pickle'))
        yearly_ = pd.read_pickle(os.path.join(loc, f'{name}_yearly.pickle'))
        error_inc = pd.read_pickle(os.path.join(loc, f'{name}_error_incremental.pickle'))
        error_cum = pd.read_pickle(os.path.join(loc, f'{name}_error_cumulative.pickle'))

    else:

        dts = load_dts()

        print(f'loading data from {workspace}')
        # get lst based water budget for ENTIRE model area
        mf_list = flopy.utils.MfListBudget(os.path.join(workspace, 'output', 'sv_model_grid_6layers.lst'))
        incremental_, cumulative = mf_list.get_dataframes(start_datetime="11-30-1969", diff=True)
        error_inc = incremental_.loc[:,['in-out', 'total', 'percent_discrepancy', 'storage']]
        error_cum = cumulative.loc[:, ['in-out', 'total', 'percent_discrepancy', 'storage']]
        lst_cum = cumulative.drop(columns=['in-out', 'total', 'percent_discrepancy', 'storage']).sum(axis=1) / 43560.
        lst_cum.index = arich_functions.water_year(lst_cum.index)
        lst_cum = lst_cum.groupby(level=0).last()
        lst_cum.index = pd.to_datetime(lst_cum.index, format='%Y')

        cum_wy = cumulative.copy()
        cum_wy.index = arich_functions.water_year(cum_wy.index)
        cum_wy = cum_wy.groupby(level=0).last() / 43560.
        cum_wy.index = pd.to_datetime(cum_wy.index, format='%Y')

        # incremental, _ = mf_list.get_dataframes(start_datetime="11-30-1969", diff=True)
        incremental_ = pd.merge(incremental_, dts.loc[:, ['date', 'ndays']], left_index=True, right_on='date')

        incremental_ = incremental_.set_index('date')
        incremental_ = incremental_.multiply(incremental_.loc[:, 'ndays'], axis='index').divide(43560.)

        yearly_ = incremental_.groupby(arich_functions.water_year(incremental_.index)).sum()
        yearly_.index = pd.to_datetime(yearly_.index, format='%Y')


        cum_wy.to_pickle(os.path.join(loc, f'{name}_cum_wy.pickle'))
        incremental_.to_pickle(os.path.join(loc, f'{name}_incremental.pickle'))
        yearly_.to_pickle(os.path.join(loc, f'{name}_yearly.pickle'))
        error_inc.to_pickle(os.path.join(loc, f'{name}_error_incremental.pickle'))
        error_cum.to_pickle(os.path.join(loc, f'{name}_error_cumulative.pickle'))

    return cum_wy, incremental_, yearly_, error_inc, error_cum


if __name__ == "__main__":
    workspace = r'C:\GSP\sv\model\SV_mod_V2\master'
    owhm2 = r'C:\GSP\sv\model\SV_mod_V2_owhm2\master'

    test_name = 'subcatv8_50percuzfPrecp_fieswi_fieswp_finf1pt0_nwt_ss_redo_finfpet'
    note = '''subcat v8. 50percuzfPrecp_fieswi_fieswp_finf1pt0.\nincreased nwt maxiter to 1000, decreasd fluxtol to 500.\n
    specific storage increased to 1e-6 where it was below.
    updated subcatchment extrapolation by adding elevation values to regression
    '''
    folder = 'versions'
    hard_load = True

    fold = setup(folder,test_name, note= note)

    plot_converge_issues(folder, owhm2, test_name, )

    plot_by_crop(folder, owhm2, test_name)

    plot_hydros(workspace, test_name, True, folder)
    plot_hydros(owhm2, test_name, False, folder)
    plot_head_map(workspace, test_name, False)

    dts = load_dts()
    cum_wy_base, incremental, yearly, error_inc,error_cum = load_bud(workspace, folder, test_name,
                                                                     base= True, hard_load = False)
    cum_wy_owhm2, incremental_owhm2, yearly_owhm2, error_owhm2_inc, error_owhm2_cum = load_bud(owhm2, folder, test_name,
                                                                                               base= False, hard_load = hard_load)

    fb_det_base,fb_raw_base = owhm.owhm.load_fb_details(os.path.join(workspace,  'FB_DETAILS.OUT'),
                                                          start_date="11-30-1969")
    fb_det_owhm,fb_owhm_base = owhm.owhm.load_fb_details(os.path.join(owhm2, 'output', 'FB_DETAILS.OUT'),
                                            start_date = "11-30-1969")

    comp_fb_details(folder, test_name, fb_raw_base, fb_owhm_base)

    plot_fb_details(folder, test_name,  fb_det_owhm, fb_owhm_base, base = False)
    plot_fb_details(folder, test_name, fb_det_base, fb_raw_base, base=True)

    bar_comp(folder, test_name, yearly, yearly_owhm2, )
    plot_stacked_bar(folder, test_name, yearly, yearly_owhm2)

    plot_ts(folder,test_name, yearly,yearly_owhm2,)

    plot_compare_q(folder, test_name, yearly, yearly_owhm2, owhm2)

    plot_error(folder, test_name, error_inc, error_cum, error_owhm2_inc, error_owhm2_cum)

    # print(f'<a href="{fold}">example text</a>')
    print("Done")




