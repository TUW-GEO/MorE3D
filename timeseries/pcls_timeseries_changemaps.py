# %%
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import geopandas as gpd # for shapefile reading and plotting
import cmcrameri as cm # for colormaps

# %%
# set parameters same as in run_levelsets.py
intvl = 24  # interval in epochs (temporal subsampling, set to 1 if all epochs should be used)
k1_full = 24  # cue 1: change in last k1 epochs
k2_full = 168  # cue 2: change in last k2 epochs
k1 = k1_full // intvl  # use epoch `n-k1` and `n-k2`
k2 = k2_full // intvl

# opals format file required for alpha shape computation with OpalsBounds
opals_fmt = '../../iformat_levelsets.xml'
# check if the file exists, if not exit
if not os.path.exists(opals_fmt):
    raise FileNotFoundError(f'{opals_fmt} does not exist, but is required by this script.')

# path to the level sets output directory
basename = f'../../kijkduin_4dobc_full_k24_168'
# check if the directory exists, if not exit
if not os.path.exists(basename):
    raise FileNotFoundError(f'{basename} does not exist, specify correct path to the data directory.')

ilist = [os.path.basename(os.path.dirname(d)) for d in glob.glob(basename+"/change_*/")]

import re
def natural_sort(input_list: list[str]) -> list[str]:
    def alphanum_key(key):
        return [int(s) if s.isdigit() else s.lower() for s in re.split("([0-9]+)", key)]
    return sorted(input_list, key=alphanum_key)

ilist = natural_sort(ilist)

# directory for the figure files
anim_dir = f'/animation'
# check if the directory exists, if not create it
if not os.path.exists(basename + anim_dir):
    os.mkdir(basename + anim_dir)

# plot the delineation on top of change maps
n = 999 # use all outputs

# add the main code dir to path, so that we can import the functions created there
import sys
from pathlib import Path
sys.path.insert(0, str((Path.cwd() / "..").resolve()))
import levelsets_func as pcls
in_file = '../../kijkduin_4dobc_full.zip'

pcdata = pcls.read_py4dgeo(in_file)

# we can reduce the data volume by the selected time interval
# TODO this only works for py4dgeo data structure, but having the time information is very useful
timedeltas_intvl = pcdata['timedeltas'][::intvl]
timedeltas_samples = np.arange(len(pcdata['timedeltas'])).astype(int)[::intvl]
for t in range(len(pcdata['timedeltas'])):
    if not t in timedeltas_samples:
        del pcdata[f'change_{t}']

pccoords = pcdata['xyz'] + pcdata['origin']

cmap_gradient = cm.cm.broc

for i in range(0,len(ilist[:n]),2): # we use every second, because there is always pos and neg
    # plt.figure(figsize=(9, 9))
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # get the change data
    change_i = int(ilist[i].split('_')[1])
    changedata_k = pcdata[f'change_{change_i}']
    changedata_grad1 = pcdata[f'change_{change_i}'] - pcdata[f'change_{change_i - k1_full}']  # pcdata[f'change_{i-1}']
    changedata_grad2 = pcdata[f'change_{change_i}'] - pcdata[f'change_{change_i - k2_full}']  # pcdata[f'change_{i-k}']

    # plot the changes
    ax1, ax2, ax3 = axs

    sc = ax1.scatter(pccoords[:, 0], pccoords[:, 1], c=changedata_k, cmap='RdYlBu_r', s=1, rasterized=True, vmin=-.5, vmax=.5)
    ax1.set_title(f'Changes day d={change_i}')

    sg = ax2.scatter(pccoords[:, 0], pccoords[:, 1], c=changedata_grad1, cmap=cmap_gradient, s=1, rasterized=True, vmin=-.2, vmax=.2)
    ax2.set_title(f'Changes gradient t-{k1_full}')

    ax3.scatter(pccoords[:, 0], pccoords[:, 1], c=changedata_grad2, cmap=cmap_gradient, s=1, rasterized=True, vmin=-.2, vmax=.2)
    ax3.set_title(f'Changes gradient t-{k2_full}')

    for change_dir in ['positive', 'negative']:
        txts = glob.glob(basename + "/" + '_'.join(ilist[i].split('_')[:-1]) + '_' + change_dir + f"/{'[0-9]' * 4}.txt")
        txts = [t.replace('\\', '/') for t in txts]
        txts = [t.replace('//', '/') for t in txts]
        itrs = [int(txt.split('/')[-1][:4]) for txt in txts]
        txt = txts[np.argmax(itrs)]
        print(txt)

        # derive the alpha shape using opalsBounds
        txt_extract = txt.replace('.txt', '_extract.txt')
        odm = txt_extract.replace('.txt', '.odm')
        shpfile = txt_extract.replace('.txt', '_bounds.shp')
        if not os.path.isfile(shpfile):
            data = np.loadtxt(txt)
            cmd_import = f'opalsImport -inf {txt_extract} -outf {odm} -iformat {opals_fmt}'
            os.system(cmd_import)
            cmd_bounds = f'opalsBounds -inf {odm} -outf {shpfile} -boundsType alphashape -alpharadius 0.5'
            os.system(cmd_bounds)

        # read the alpha shape from shapefile
        bounds = gpd.read_file(shpfile)

        bound_col = 'red' if change_dir == 'pos' else 'blue'
        for ax in axs:
            # plot the alpha shape
            bounds.plot(ax=ax, color='none', edgecolor=bound_col, linewidth=0.5)
            ax.set_xlim([-350, 0])
            ax.set_ylim([-300, 350])
            ax.set_aspect('equal')

    plt.axis('equal')
    plt.colorbar(sc, label='Height change [m]',ax=ax1)
    plt.colorbar(sg, label='Change gradient [m]',ax=ax3)

    plt.savefig(basename + f'{anim_dir}/{change_i:03d}b.png', dpi=196)
    plt.close()


