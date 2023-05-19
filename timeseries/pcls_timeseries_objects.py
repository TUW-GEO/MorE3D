import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import geopandas as gpd # for shapefile reading and plotting
import cmcrameri as cm # for colormaps

# add the main code dir to path, so that we can import the functions created there
import sys
from pathlib import Path
sys.path.insert(0, str((Path.cwd() / "..").resolve()))
import levelsets_func as pcls

# enable interactive plotting in debug mode
import matplotlib as mpl
#mpl.use('TkAgg')
mpl.use('Agg')

# set parameters same as in run_levelsets.py
intvl = 24  # interval in epochs (temporal subsampling, set to 1 if all epochs should be used)
k1_full = 24  # cue 1: change in last k1 epochs
k2_full = 168  # cue 2: change in last k2 epochs
k1 = k1_full // intvl  # use epoch `n-k1` and `n-k2`
k2 = k2_full // intvl

# specify the input files
basename = f'../../kijkduin_4dobc_full_k24_168' # directory with level sets output
in_file = '../../kijkduin_4dobc_full.zip' # input file with time series data (here: py4dgeo object)

# check if the result directory exists, if not exit
if not os.path.exists(basename):
    raise FileNotFoundError(f'{basename} does not exist, specify correct path to the data directory.')

ilist = [os.path.basename(os.path.dirname(d)) for d in glob.glob(basename+"/change_*/")]

import re
def natural_sort(input_list: list[str]) -> list[str]:
    def alphanum_key(key):
        return [int(s) if s.isdigit() else s.lower() for s in re.split("([0-9]+)", key)]
    return sorted(input_list, key=alphanum_key)

ilist = natural_sort(ilist)

# read the data
pcdata = pcls.read_py4dgeo(in_file)

# we can reduce the data volume by the selected time interval
timedeltas_intvl = pcdata['timedeltas'][::intvl]
timedeltas_samples = np.arange(len(pcdata['timedeltas'])).astype(int)[::intvl]
for t in range(len(pcdata['timedeltas'])):
    if not t in timedeltas_samples:
        del pcdata[f'change_{t}']

pccoords = pcdata['xyz'] + pcdata['origin']
reference_epoch = pcdata['time_reference']

gdf_file = basename + '/gdf.pkl'
if os.path.isfile(gdf_file):
    gdf = pd.read_pickle(gdf_file)

else:

    # create an empty geodataframe to store the polygons
    gdf = gpd.GeoDataFrame(columns=['id', 'pid', 'polygon', 'timedelta_int', 'neigh_t'], geometry='polygon')

    for i in range(0,len(ilist),2): # we use every second, because there is always pos and neg
        print(i, ilist[i])
        # get the change data
        change_i = int(ilist[i].split('_')[1])
        changedata_k = pcdata[f'change_{change_i}']
        changedata_grad1 = pcdata[f'change_{change_i}'] - pcdata[f'change_{change_i - k1_full}']  # pcdata[f'change_{i-1}']
        changedata_grad2 = pcdata[f'change_{change_i}'] - pcdata[f'change_{change_i - k2_full}']  # pcdata[f'change_{i-k}']

        id_counter = 0
        for change_dir in ['positive', 'negative']:
            txts = glob.glob(basename + "/" + '_'.join(ilist[i].split('_')[:-1]) + '_' + change_dir + f"/{'[0-9]' * 4}.txt")
            txts = [t.replace('\\', '/') for t in txts]
            txts = [t.replace('//', '/') for t in txts]
            itrs = [int(txt.split('/')[-1][:4]) for txt in txts]
            txt = txts[np.argmax(itrs)]

            # derive the alpha shape using opalsBounds
            txt_extract = txt.replace('.txt', '_extract.txt')
            odm = txt_extract.replace('.txt', '.odm')
            shpfile = txt_extract.replace('.txt', '_bounds.shp')
            if not os.path.isfile(shpfile):
                print(f'Bounds file "{shpfile}" does not exist. Derive it external procedure (e.g., opalsBounds in pcls_timeseries_changemaps.py')
                continue

            # read the alpha shape from shapefile
            bounds = gpd.read_file(shpfile)

            # get the timedelta of the current epoch
            timedelta_int = pcdata['timedeltas'][change_i]

            # loop over all polygons in the multipolygon and add them to the geodataframe
            for p, poly in enumerate(bounds.geometry[0]):
                gdf = gdf.append({'id': id_counter, 'pid': p, 'polygon': poly, 'timedelta_int': timedelta_int, 'neigh_t':np.nan}, ignore_index=True)
                id_counter += 1

    # convert timedelta_int to timedelta object
    gdf['timedelta'] = gdf['timedelta_int'].apply(lambda x: pd.Timedelta(seconds=x))

    # add absolute time information by converting the timedelta to seconds and adding the reference epoch
    gdf['timestamp'] = gdf['timedelta'] + reference_epoch

    # sort the geodataframe by the timestamp (descending, because we want to start with the most recent epoch in operational monitoring)
    gdf = gdf.sort_values(by='timestamp', ascending=False, ignore_index=True)

    # ensure that we are still working with a geodataframe
    gdf = gpd.GeoDataFrame(gdf, geometry='polygon')

    # save the geodataframe to file
    gdf.to_pickle(gdf_file)

# add seed information to the geodataframe
gdf['status'] = 'candidate' # will be updated to 'seed' or 'neigh' later

# loop over all polygons and find the temporal connections (space-time neighbors)

iou_thr = 0.5 # intersection over union threshold for the polygons
area_thr = 25.0 # area threshold for the polygons

# for testing, we select only polygons that cover a certain location
# using a shapely point object
from shapely.geometry import Point
# can be one point or GDF; set to None to analyze all the entire scene
locs_sel = None #gpd.GeoDataFrame({'points':[Point(-250.0, 0.0), Point(-200.0, 0.0)]}, geometry='points')

# select the polygons that contain the selected points point
if not locs_sel is None:
    gdf = gdf.sjoin(locs_sel, predicate='contains', how='inner')
    # rename column 'index_right' to 'locs_sel'
    gdf = gdf.rename(columns={'index_right':'locs_sel'})

gdf_pkl = f'{basename}/gdf.pkl'

# if the gdf_pkl file exists, load it
if os.path.isfile(gdf_pkl):
    gdf = gpd.read_pickle(gdf_pkl)

# if the gdf_pkl file does not exist, create it
else:

    # ensure that all polygons are valid
    from shapely.validation import make_valid
    gdf['polygon'] = [make_valid(p) for p in gdf['polygon']]

    # reset index of gdf for iterating
    gdf = gdf.reset_index(drop=True)
    i=0
    oid = 0
    num_poly = len(gdf)
    while i < len(gdf):

        # output status to console to track progress
        print(f'Processing polygon {i+1}/{len(gdf)} ({int(round(((float(i)+1)/num_poly*100), 0))}%)')

        seed_curr = gdf.iloc[i]

        # continue if polygon area is too small
        if seed_curr['polygon'].area < 50:
            i += 1
            continue

        # if the polygon is already connected to another polygon, skip it
        if seed_curr['status'] in ['seed', 'neigh']:
            i += 1
            continue

        # update the status of the seed polygon
        gdf.loc[i, 'status'] = 'seed'
        # assign an object id to the seed
        gdf.loc[i, 'oid'] = oid

        # find all polygons that are within the temporal window (but not in the same epoch)
        gdf_neigh = gdf[(gdf['timestamp'] < seed_curr['timestamp'] - pd.Timedelta(seconds=k1_full*3600))]

        # remove polygons that are already connected to another polygon (neigh_t is not NaN)
        gdf_neigh = gdf_neigh[gdf_neigh['neigh_t'].isna()]

        # if there is no neighbor, skip the polygon
        if len(gdf_neigh) > 0:

            # find all polygons that have spatial overlap with the current polygon and derive the intersection and union
            gdf_neigh['intersects'] = gdf_neigh['polygon'].apply(lambda x: seed_curr['polygon'].intersection(x).area)
            gdf_neigh['unions'] = gdf_neigh['polygon'].apply(lambda x: seed_curr['polygon'].union(x).area)

            # calculate the intersection over union (IoU) for all polygons
            gdf_neigh['ious'] = gdf_neigh['intersects'] / gdf_neigh['unions']

            # select the polygon with the highest IoU per epoch, but only if not already connected to another polygon and if the IoU is above the threshold
            gdf_neigh = gdf_neigh[(gdf_neigh['neigh_t'].isna()) & (gdf_neigh['ious'] > iou_thr)]

            # if there is no neighbor left, continue with the next seed polygon
            if len(gdf_neigh) > 0:
                # now we found a temporal sequence of polygons connected to one seed polygon
                # assign the polygon id of the later polygon (t+1) as neigh_t for each polygon
                gdf_neigh['neigh_t'] = gdf_neigh['id'].shift(1)

                # the first polygon in the sequence has no neigh_t, so we set it to the id of the seed polygon
                gdf_neigh.iloc[0, gdf_neigh.columns.get_loc('neigh_t')] = seed_curr['id']

                # update the neigh_t column in the geodataframe
                gdf.loc[gdf_neigh.index, 'neigh_t'] = gdf_neigh['neigh_t']
                gdf.loc[gdf_neigh.index, 'status'] = 'neigh'

                # update the object id of the neigh_t polygons
                gdf.loc[gdf_neigh.index, 'oid'] = oid

        if 0:
            # visualize the sequence
            fig, ax = plt.subplots(figsize=(10,10))

            # get the change data for the current seed polygon
            change_i = np.where(pcdata['timedeltas']==seed_curr['timedelta'].total_seconds())[0][0]
            changedata_curr = pcdata[f'change_{change_i}']
            ax.scatter(pccoords[:,0], pccoords[:,1], c=changedata_curr, cmap='RdYlBu_r', s=1, rasterized=True, vmin=-.5, vmax=.5)

            # plot the seed polygon
            gdf.loc[[i], 'polygon'].plot(ax=ax, edgecolor='blue', facecolor='none', lw=2)

            # plot the selected point location
            # ax.scatter(locs_sel.x, locs_sel.y, c='red', s=5, label='point location for spatial selection')

            # plot the temporal sequence of polygons
            if len(gdf_neigh) > 0:
                gdf_neigh_ts = np.nan
                for j, jr in gdf_neigh.iterrows():
                    # ax.plot(gdf_neigh.iloc[j]['polygon'].exterior.xy, c='b', lw=2)
                    gdf_neigh.loc[[j], 'polygon'].plot(ax=ax, edgecolor='black', alpha=0.3, facecolor='none', lw=0.5)
                    gdf_neigh_ts = jr['timestamp']
                plt.suptitle(f'Polygon {seed_curr["id"]} (duration: {int((seed_curr["timestamp"] - gdf_neigh_ts).total_seconds()/3600)} h)')
                plt.title(f'{gdf_neigh_ts} to {seed_curr["timestamp"]}')
            else:
                plt.suptitle(f'Polygon {seed_curr["id"]}')
                plt.title(f'{seed_curr["timestamp"]}')

            # set equal aspect ratio
            ax.set_aspect('equal')

            # save the figure
            fig.tight_layout()
            plt_dir = f'{basename}/plots'
            if not os.path.exists(plt_dir):
                os.makedirs(plt_dir)
            plt.savefig(f'{plt_dir}/epoch_{change_i}_seed_{seed_curr["id"]}.png', dpi=200)
            plt.close()

        gdf_neigh = None
        i+=1
        oid+=1

    # save the geodataframe
    gdf.to_pickle(gdf_pkl)

