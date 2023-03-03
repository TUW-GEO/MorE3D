'''
===================================================================
                            MorE3d
===================================================================

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


'''

import numpy as np
import levelsets_func as pcls
import os
from multiprocessing import Pool

def process(data, args, base_dir, restrict_domain=''):
    field1, field2, field3, parameters = args

    # re-use intermediate calculations (neighbors, normals, tangents)
    reuse_intermediate = True

    # active contours: Chan-Vese (`chan_vese`) or Local Mean and Variance ('lmv')
    active_contour_model = 'chan_vese'

    # each cycle runs a number of steps then stores a result
    num_cycles = 12
    num_steps = 50

    # number of smoothing passes for zeta
    num_smooth = 1

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

    # robust cues, clip at X%
    cue_clip_pc = 99.9

    # initialization voxel size
    vox_size = 10

    # initialization cue percentage
    init_pc = 50

    # initialization method (voxel/cue)
    init_method = 'voxel'

    # neighbor threshold for points to be extracted
    # (must have >= salient neighbors to be extracted)
    extraction_threshold = k // 2

    # recenter cues by substracting cue median
    center_data = False

    for key in parameters.keys():
        vars()[key] = parameters[key]

    print(f"processing '{field1}'/'{field2}' | restriction: {restrict_domain}")

    out_dir = os.path.join(base_dir, f"{field1}_{field2}")
    if restrict_domain is not None:
        out_dir += f"_{restrict_domain}"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    points = data["xyz"]

    zeta = np.zeros((points.shape[0], 2))
    zeta[:, 0] = data[field1].copy()

    if field2 is not None:
        zeta[:, 1] = data[field2].copy() - data[field1].copy()

    zeta[np.isnan(zeta)] = 0

    if center_data:
        zeta[:, 0] -= np.median(zeta[:, 0])
        zeta[:, 1] -= np.median(zeta[:, 1])
    if restrict_domain == 'positive':
        zeta[:, 0][zeta[:, 0] < 0] = 0
        zeta[:, 1][zeta[:, 1] < 0] = 0
    elif restrict_domain == 'negative':
        zeta[:, 0][zeta[:, 0] > 0] = 0
        zeta[:, 1][zeta[:, 1] > 0] = 0

    tmp_file = os.path.join(base_dir, 'tmp.npz')
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

if __name__ == '__main__':

    # --- config / parameters ---
    pcls.verbose = False
    pcls.checks = False

    # - multi features
    # - added feature: some kind of temporal change info

    intvl = 24  # interval in epochs (temporal subsampling, set to 1 if all epochs should be used)
    k1_full = 24 # cue 1: change in last k1 epochs
    k2_full = 168 # cue 2: change in last k2 epochs
    k1 = k1_full//intvl   # use epoch `n-k1` and `n-k2`
    k2 = k2_full//intvl

    # data
    in_file = 'kijkduin_4dobc_full.zip' # py4dgeo object

    # field to use as cue
    base_dir = os.path.join(os.path.dirname(in_file),
                            os.path.splitext(os.path.basename(in_file))[0]+f'_k{k1_full}_{k2_full}')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # --- run ---
    data = pcls.read_py4dgeo(in_file)
    fields = data['fields']

    # we can reduce the data volume by the selected time interval
    # TODO this only works for py4dgeo data structure, but having the time information is very useful
    timedeltas_intvl = data['timedeltas'][::intvl]
    timedeltas_samples = np.arange(len(data['timedeltas'])).astype(int)[::intvl]
    for t in range(len(data['timedeltas'])):
        if not t in timedeltas_samples:
            del data[f'change_{t}']

    v_ = vars()
    v = {key: v_[key] for key in v_.keys() if
         key[0] != '_' and isinstance(v_[key], (int, bool, str, float))}

    # use multiprocessing to parallelize
    import multiprocessing as mp
    import time
    import sys

    numthreads = 10

    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        pass
    elif gettrace():
        print('Debugging mode detected, running only single thread mode')
        numthreads = 1

    procs = []
    for _ in zip(fields[::intvl][k2::], fields[::intvl][k2 - k1:-k1], fields[::intvl][:-k2], [v] * (len(fields[::intvl]))):
        for changedir in ['positive', 'negative']:
            proc = mp.Process(target=process, args=(data, _, base_dir, changedir))
            procs.append(proc)

    last_started_idx = -1
    running_ps = []
    while True:
        if len(running_ps) < numthreads:
            last_started_idx += 1
            if last_started_idx < len(procs):
                procs[last_started_idx].start()
                running_ps.append(last_started_idx)

        for running_p in running_ps:
            # see if there is a process (thread) that is terminated
            if not procs[running_p].is_alive():
                # close the terminated process
                procs[running_p].close()
                running_ps.remove(running_p)

        time.sleep(1)

        if last_started_idx >= len(procs) and len(running_ps) == 0:
            break
