"""
@Created: Reuma Arav
@Date: January 2022

Script for downsampling point cloud based on saliency and using the voxel based method (as implemented in open3d)


=================================================================
MIT License

Copyright (c) 2022 TU Wien - Department of Geodesy and Geoinformation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=======================================================================


"""
from BallTreePointSet import BallTreePointSet
import open3d as o3d
from skimage.filters import threshold_otsu

import numpy as np

if __name__ == '__main__':
    folder = '../data/Zeelim'
    saliency_folder = folder + '/saliency/'
    fname = 'Zeelim1.5M.txt'
    leaf_size = 40
    zero_thresh = 1e-4
    saliency = np.loadtxt(saliency_folder + fname)
    pts = saliency[:, :3]
    btP = BallTreePointSet(pts, leaf_size=leaf_size)

    # check the level in the required LOD (radius)
    lod = [1, 2, 5, 7, 10] # airborne datasets
    # lod = [0.05, 0.1, 0.5, 1] # terrestrial datasets

    for LOD in lod:
        print('computing {}'.format(LOD))

        smallest_nodes_idx = btP.getSmallestNodesOfSize(LOD)

        new_pts = []
        salient_flag = []
        segmentation = []
        labels = np.zeros(btP.Size)
        # for each sibling, check if the points therein are saliency or not.
        s_thresh = threshold_otsu(saliency[:, 3]) # use Otsu's thresholding

        for i in smallest_nodes_idx:
            node_idx = btP.getPointsOfNode(i)
            idx_saliency = saliency[node_idx, 3] > s_thresh and np.abs(saliency[node_idx, 3]) < zero_thresh

            if np.sum(idx_saliency) * 100 / idx_saliency.shape[0] > 60:
                # keep cell as is
                pts_tmp = pts[node_idx, :3]
                salient_tmp = np.ones(pts_tmp.shape[0])

            else:
                # keep salient points as is and average non-salient
                average_pt = np.mean(pts[node_idx][1 - idx_saliency], axis=0)

                # keep the closest original point to the averaged one
                average_pt = btP.query(average_pt[None, :],1)
                pts_tmp = np.vstack((pts[node_idx][idx_saliency], btP.GetPoint(average_pt[0][0])))

                # add a property of 1 for salient points 0 for non-salient
                salient_tmp = np.zeros(pts_tmp.shape[0])
                salient_tmp[:pts[node_idx][idx_saliency].shape[0]] = 1
            for p, s in zip(pts_tmp, salient_tmp):
                new_pts.append(p)
                salient_flag.append(s)

        # save results
        saliency_down = np.asarray(new_pts)
        salient_flag = np.asarray(salient_flag)
        np.savetxt(folder + '/downsampled/' + fname[:-3] + '_' + str(LOD) + '_' + 'saliency.txt', np.hstack((saliency_down, salient_flag[:, None])))

        # ------------ downsample by voxel grid ----------------
        p3d = o3d.geometry.PointCloud()
        p3d.points = o3d.utility.Vector3dVector(pts[:, :3])
        voxel_down = p3d.voxel_down_sample(voxel_size=LOD)
        voxel_down = np.asarray(voxel_down.points)
        print('number of points', str(voxel_down.shape[0]))
        np.savetxt(folder + '/downsampled/' + fname[:-3] + '_' + str(LOD) + '_voxel.txt', voxel_down)

