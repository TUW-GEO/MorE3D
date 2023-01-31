"""
===================================================================
                            MorE3d
===================================================================
@Created: Reuma Arav
@Date: January 2022

Script for comparing results of different downsampling datasets

When using this code please cite:

@article{Arav2022content,
  title={Content-Aware Point Cloud Simplification of Natural Scenes},
  author={Arav, Reuma and Filin, Sagi and Pfeifer, Norbert},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--12},
  year={2022},
  publisher={IEEE}
}


"""
from sklearn.neighbors import BallTree

import open3d as o3d
import numpy as np
import itertools

if __name__ == '__main__':
    folder = '../data/Zeelim/'
    saliency_folder = folder + 'saliency/'
    fname = 'Zeelim1.5M.txt' # these data can be found on https://doi.org/10.48436/mps0m-c9n43

    pts = np.loadtxt(saliency_folder + fname)

    leafsize = 40 # leaf size for ball tree reconstruction
    knn = 64
    rnn = .1

    rnn_flag = True # if true, run radius based neigbourhood
    upwards_z = True # if true, normals with negative Z are reversed (not for full 3D datasets).

    btP = BallTree(pts[:, :3], leafsize)

    # used LOD (radius)
    # LOD = [1, 2, 5, 10] # Kaunertal and Zeelim
    # LOD = [1, 0.5, 0.1, 0.05] # Leopards and cave
    LOD = [1.15]
    # ------------------ load subsampled by other software ----------
    normals_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[:, :3]))

    if rnn_flag: # for radius based neighbourhood
        normals_pts.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(rnn))
    else: # for the knn based neighbourhood
        normals_pts.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn))

    n = np.asarray(normals_pts.normals)

    if upwards_z:
        # make sure Z is upwards
        n[n[:, 2] < 0] = -n[n[:, 2] < 0]

    normals_pts.normals = o3d.utility.Vector3dVector(n)

    # initialize lists for saving results
    E_c2c = []
    E_c2p = []
    dN = []
    dataset_list = []
    E_all = []
    dN_all = []

    for lod in LOD:
        E_c2c_tmp = []
        E_c2p_tmp = []
        dN_tmp = []
        E_tmp = []
        dN_all_tmp2 = []

        saliency_down = np.loadtxt(folder + '/downsampled/' + fname[:-3] + '_' + str(10) + '_saliency.txt')
        mesh_down = np.loadtxt(folder + 'downsampled/' + fname[:-3] + '_'+ str(0.875) +'_mesh.xyz')
        curvature_down =  np.loadtxt(folder + 'downsampled/' + fname[:-3] +'_'+ str(1.15) + '_curvature.txt')
        voxel_down = np.loadtxt(folder + 'downsampled/' + fname[:-3] + '_'+str(1.35) + '_voxel.txt')

        if curvature_down.shape[1] > 3:
            curvature_down = curvature_down[:, :3]

        if mesh_down.shape[1] > 3:
            mesh_down = mesh_down[:, :3]

        if voxel_down.shape[1] > 3:
            voxel_down = voxel_down[:, :3]

        if saliency_down.shape[1] > 3:
            saliency_down = saliency_down[:, :3]

        dataset_list_tmp = [ saliency_down, voxel_down, mesh_down, curvature_down]
        dataset_names = ['saliency',  'voxel', 'mesh', 'curvature']

        # --------------- Compare results -------------
        for downsampled, name in zip(dataset_list_tmp, dataset_names):
            print(name + str(lod))
            # compute normals
            downsampled_bt = BallTree(downsampled[:, :3], leafsize)
            normals = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(downsampled[:, :3]))

            try:
                if rnn_flag:
                    normals.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamRadius(rnn))
                else:
                    normals.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamKNN(knn)) # for the rest

            except: # in case there are not enough neighbours
                knn = int(downsampled.shape[0]/2)
                normals.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamKNN(knn))
            normals.normalize_normals()

            n = np.asarray(normals.normals)

            if upwards_z:
               # make sure Z is upwards
                n[n[:, 2] < 0] = -n[n[:, 2] < 0]

            normals.normals = o3d.utility.Vector3dVector(n)

            # estimate cloud-to-cloud and point-to-plane error (Tian et al. 2017)
            # 1. distance between each point to its closest -- downsampled distance from original
            E, closest = btP.query(downsampled[:,:3], 1)
            closest = closest.flatten()
            vector_E = pts[closest, :3] - downsampled[:,:3]
            # 2. distance between each point to its closest -- original distance from downsampled
            E2, closest2  = downsampled_bt.query(pts[:, :3], 1)
            closest2 = closest2.flatten()
            vector_E2 = pts[:,:3] - downsampled[closest2, :3]

            E_tmp.append(np.average(E2) + np.average(E) / 2)
            E_c2c_tmp.append(np.average(E2) + np.average(E)/2)  # cloud-to-cloud

            E_c2p_tmp.append(np.average(np.einsum('ij,ij->i',vector_E2, np.asarray(normals_pts.normals)[closest2, :]) ** 2))  # cloud2plane
            dn_all_tmp = np.einsum('ij,ij->i', np.asarray(normals_pts.normals)[closest, :] , np.asarray(normals.normals)) # angle between normals
            dn_all_tmp[np.abs(dn_all_tmp) > 1] = 1  # numerical correction
            dn_all_tmp = np.degrees(np.arccos(dn_all_tmp))
            dn_all_tmp[dn_all_tmp > 90] = 180 - dn_all_tmp[dn_all_tmp > 90]

            dN_tmp.append(np.median(dn_all_tmp))  # normal difference
            dN_all_tmp2.append(dn_all_tmp)

    # save and print results
        E_all.append(E_tmp)
        dN_all.append(dN_all_tmp2)
        E_c2c.append(E_c2c_tmp)
        E_c2p.append(E_c2p_tmp)
        dN.append(dN_tmp)
        dataset_list.append(dataset_list_tmp)
    print('_' * 150)
    print('lod \t name \t num  \t c2c \t c2p \t dN ')
    print('-' * 50)
    data_iter = itertools.cycle(dataset_list)
    dataset_names_iter = itertools.cycle(dataset_names)
    for lod, ec2c, ec2p, n, d, e in zip(LOD, E_c2c, E_c2p, dN, dataset_list, dN_all):
        for  c2c, c2p, dn, data, errors in zip( ec2c, ec2p, n, d, e):
            name = dataset_names_iter.__next__()
            print(str(lod) + '\t' + name+ '\t' + str(data.shape[0]) + '\t' + str(c2c) + '\t' + str(c2p) + '\t' + str(dn) )
            try:
                np.savetxt(folder + '/downsampled/' + fname[:-4] + '_' + str(lod) + '_' + name + '_E.txt',
                       np.hstack((data, errors)))
            except:
                np.savetxt(folder + '/downsampled/' + fname[:-4] + '_' + str(lod) + '_' + name + '_E.txt',
                           np.hstack((data, errors[:,None])))


