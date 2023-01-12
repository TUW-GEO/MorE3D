'''
Main to run the Point-based level-set for the extraction of 3D entities
-----------------------------------------------------------------------

@author: Florian Poeppl
@date: 1/7/2022

# Dependencies:
    - sklearn

When using this code please cite:

@article{Arav.etal2022,
  title={A POINT-BASED LEVEL-SET APPROACH FOR THE EXTRACTION OF 3D ENTITIES FROM POINT CLOUDS--APPLICATION IN GEOMORPHOLOGICAL CONTEXT},
  author={Arav, R. and P{\"o}ppl, F. and Pfeifer, N.},
  journal={ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume={2},
  pages={95--102},
  year={2022},
  publisher={Copernicus GmbH}
}

---------------------------------------------------------------------
-                               MorE3D
---------------------------------------------------------------------
The MIT License (MIT)

Copyright 2022, Photogrammetry Research Group, TU Wien

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

-----------------------------------------------------------------------------

'''

import numpy as np
import levelsets_func as pcls
import os
from argparse import Namespace
from multiprocessing import Pool

# --- config / parameters ---
pcls.verbose = False
pcls.checks = True

# - multi features
# - added feature: some kind of temporal change info

#  in_file = './datasets/Zeelim_sub05m.las'  
#  field = 'saliency'  

#  in_file = './datasets/schneeferner_m3c2_sel_days_aoi2_sub025.las'   
#  field = 'm3c2_180422_120031'   
in_file = './datasets/beach/change_timeseries_tint24_nepochs123.laz'
#  in_file = './datasets/snowcover/change_timeseries_tint1_nepochs129_subsampled1.las'   
fields = [f'change_{i}' for i in range(0, 123)]
k = 5  # use epoch `n` and `n+k`

# re-use intermediate calculations (neighbors, normals, tangents)
reuse_intermediate = True

# active contours: Chan-Vese (`chan_vese`) or Local Mean and Variance ('lmv')
active_contour_model = 'chan_vese'

# each cycle runs a number of steps then stores a result
num_cycles = 6
num_steps = 50

# number of smoothing passes for zeta
num_smooth = 5

# explicit euler stepsize
stepsize = 1000

# controls regularization
nu = 0.0001

# controls curvature
mu = 0.0025

# controls zeta-in term
lambda1 = 1.0

# controls zeta-out term
lambda2 = 1.0

# heaviside/delta approximation "width", is scaled with h
epsilon = 1.0

# approximate neighborhood radius
h = 2.5  # (all k neighbours should be within)

# number of kNN neighbors
k = 7

# termination tolerance
tolerance = 5e-5

# field to use as cue
base_dir = os.path.join(os.path.dirname(in_file), 
                        os.path.splitext(os.path.basename(in_file))[0])
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# robust cues, clip at X%
cue_clip_pc = 97.50

# initialization voxel size
vox_size = 10

# initialization cue percentage
init_pc = 50

# initialization method (voxel/cue)
init_method = 'voxel'

# neighbor threshold for points to be extracted
# (must have >= salient neighbors to be extracted)
extraction_threshold = k//2

# --- run ---

print(f"reading file '{in_file}'")
data = pcls.read_las(in_file, fields)
tmp_file = os.path.join(base_dir, 'tmp.npz')

v_ = vars()
v = {key: v_[key] for key in v_.keys() if 
        key[0] != '_' and isinstance(v_[key], (int, bool, str, float))}

fields12 = zip(fields[:-k], fields[k:])

def process(args):
    field1, field2, parameters = args

    for key in parameters.keys():
        vars()[key] = parameters[key]

    print(f"processing '{field1}'/'{field2}'")

    out_dir = os.path.join(base_dir, f"{field1}_{field2}")

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    points = data["xyz"]

    zeta = np.zeros((points.shape[0], 2))
    zeta[:, 0] = data[field1].copy()
    zeta[:, 1] = data[field2].copy() - data[field1].copy()

    if reuse_intermediate and os.path.exists(tmp_file):
        # load neighborhoods, normals, tangents
        #  print('loading previous neighbors/normals/tangents') 
        archive = np.load(tmp_file)
        neighbors = archive['neighbors']
        normals = archive['normals']
        tangents = archive['tangents']
        if (normals.shape[0] != points.shape[0] or
            neighbors.shape[0] != points.shape[0] or
            neighbors.shape[1] != k):
            #  print('neighborhood changed, recomputing') 
            os.remove(tmp_file)
    if not reuse_intermediate or not os.path.exists(tmp_file):
        # compute neighborhoods, normals, tangents
        neighbors, normals, tangents = pcls.build_neighborhoods(points, h, k)
        np.savez_compressed(tmp_file,
                            neighbors=neighbors, normals=normals, tangents=tangents)


    # build MLS approximations
    solver = pcls.Solver(h, zeta, points, neighbors, tangents,
                         active_contour_model)

    for i in range(num_smooth):
        solver.zeta = pcls.smooth(solver.zeta, slice(None), solver.neighbors, solver.diff.weights)

    # normalize field to [0, 1] & cut off the long tail
    solver.zeta = np.abs(solver.zeta)
    if np.sum(np.isnan(solver.zeta)) > 0:
        solver.zeta[np.isnan(solver.zeta)] = 0
    pcls.clip(solver.zeta, 0, np.percentile(solver.zeta, cue_clip_pc))
    pcls.normalize(solver.zeta, 1)


    if init_method == 'voxel':
        solver.initialize_from_voxels(vox_size)
    if init_method == 'cue':
        solver.initialize_from_zeta(init_pc)

    solver.phi[:] = pcls.smooth(
            solver.phi, slice(None), solver.neighbors, solver.diff.weights
        )

    step = 0

    N = num_cycles
    M = num_steps

    solver.save(os.path.join(out_dir, f'{step:04d}'), data['origin'])

    converged = False
    for i in range(N):
        #  print('cycle', i+1, 'step', step) 
        tol = 0 if i == 0 else tolerance
        converged = solver.run(M, stepsize=stepsize, nu=nu, mu=mu, tolerance=tol,
                lambda1=lambda1, lambda2=lambda2, epsilon=epsilon, cap=True)
        step += M
        solver.save(os.path.join(out_dir, f'{step:04d}'), data['origin'], 
                extract=True, extraction_threshold=extraction_threshold) 
        if converged:
            #  print('convergence!') 
            break

    # ---


pool = Pool()


for _ in zip(fields[:-k], fields[k:], [v]*len(fields)):
    process(_)

#  pool.map(process, zip(fields, [v]*len(fields))) 

