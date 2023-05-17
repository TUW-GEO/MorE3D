'''
===================================================================
                            MorE3d
===================================================================

Functions for running a level-set method for contour extraction on point clouds.
---------------------------------------------------------------------------------
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
import scipy as sp
from scipy.spatial import Delaunay
from sklearn.neighbors import BallTree

verbose = False
def read_ply(filename):
    """read input data from ply file"""
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print('plyfile module not found (cannot load .ply files)')

    _print(f"loading data fom {filename}")
    data = {}
    plydata = PlyData.read(filename)
    data['xyz'] = np.column_stack(
        [plydata['vertex'].data[_] for _ in ('x', 'y', 'z')]).view().astype('float64')
    data['origin'] = data['xyz'].mean(axis=0)
    data['xyz'] -= data['origin']
    data['normals'] = np.column_stack(
        [plydata['vertex'].data[_] for _ in ('nx', 'ny', 'nz')]).view().astype('float64')
    data['saliency'] = np.array(
        plydata['vertex'].data['scalar_saliency']).astype('float64')
    return data


def read_las(filename, attributes=None):
    try:
        import laspy
    except ImportError:
        print('laspy module not found (cannot load .ply files)')

    las = laspy.read(filename, )
    data = {}
    data['xyz'] = np.column_stack((las.x, las.y, las.z))
    data['origin'] = data['xyz'].mean(axis=0)
    data['xyz'] -= data['origin']
    if attributes:
        for attribute in attributes:
            data[str(attribute)] = las.points.__getattr__(attribute)
    try:
        data['curvature'] = las.points.Curvature
    except:
        pass

    try:
        data['normals'] = np.column_stack(
            (las.NormalX, las.NormalY, las.NormalZ))
    except:
        pass

    try:
        data['saliency'] = las.points.saliency
    except:
        pass

    return data


def read_py4dgeo(in_file, data_pkl=None):
    """read input data from py4dgeo file

    filename: path to py4dgeo file
    data_pkl: path to pickle file containing data (optional, otherwise estimated from filename). If not existent it will be created.
    """
    import os
    import py4dgeo
    import pickle

    data_pkl = in_file.replace('.zip', '.pickle')
    if not os.path.isfile(data_pkl):
        print('reading data from py4dgeo object')
        import py4dgeo
        data = {}
        data_obj = py4dgeo.SpatiotemporalAnalysis(in_file)
        data['xyz'] = data_obj.corepoints.cloud
        data['origin'] = np.round(np.median(data['xyz'], axis=0), 0)
        data['xyz'] = data['xyz'] - data['origin']  # similar to read_las func, we also apply this offset 'globally' for the script
        data['timedeltas'] = np.array([int(dt.total_seconds()) for dt in data_obj.timedeltas])  # in seconds
        data['time_reference'] = data_obj.reference_epoch.timestamp
        distances = data_obj.distances_for_compute
        fields = [f'change_{i}' for i in range(0, len(distances[0]))]
        data['fields'] = fields
        for t in range(len(data['timedeltas'])):
            data[f'change_{t}'] = np.array([cp[t] for cp in distances])
        print('saving data to pickle file')
        with open(data_pkl, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('reading data from pickle file')
        with open(data_pkl, 'rb') as f:
            data = pickle.load(f)
    return data

def _print(string, l=0, **kwargs):
    if verbose:
        print(''.join([' '] * l) + string, **kwargs, **{'flush': True})


def wendland(x, h):
    """wendland weighting function"""
    i = x < h
    out = np.zeros_like(x)
    out[i] = (1 - x[i] / h) ** 4 * (4 * x[i] / h + 1)
    return out


def normalize(array, s=1):
    if np.any(array != 0):
        array -= array.min()
        array /= array.max()
        array *= s


def clip(array, a, b):
    array[array < a] = a
    array[array > b] = b


def compute_normals(points, neighbors):
    """compute normals via PCA of neighbors"""
    if type(neighbors) is np.ndarray and type(neighbors[0]) is np.ndarray:
        C = points[neighbors] - points[:, None, :]
        U, S, Vh = np.linalg.svd(C)
        normals = Vh[:, -1]
    else:
        normals = np.zeros_like(points)
        for i in range(points.shape[0]):
            nbs = neighbors[i]
            if not len(nbs) > 3:
                raise AssertionError
            else:
                C = points[nbs] - points[i, None, :]
                U, S, Vh = np.linalg.svd(C)
                normals[i] = Vh[(-1)]
                _print(f"{(i + 1) / points.shape[0] * 100:.2f}%", 2, end='\r')
    return normals


def compute_tangent_planes(points, neighbors, normals):
    """compute 2 orthogonal vectors spanning the tangent plane at each point"""
    N = points.shape[0]
    t1 = np.zeros((N, 3))
    t2 = np.zeros((N, 3))
    for i in range(N):
        x = points[i]
        n = normals[i]
        n /= np.linalg.norm(n)
        t0 = x - points[neighbors[i][(-1)]]
        t1[i] = t0
        t1[i] -= np.dot(n, t1[i]) * n
        t1[i] /= np.linalg.norm(t1[i])
        t2[i] = np.cross(n, t1[i])
        t2[i] /= np.linalg.norm(t2[i])
        assert abs(n @ t1[i]) < 1e-13
        assert abs(n @ t2[i]) < 1e-13
        if i % 10000 == 0:
            _print(f"{(i + 1) / points.shape[0] * 100:.2f}%", 2, end='\r')
    return t1, t2


def build_neighborhoods(points, h, k, normals=None):
    """build point neighborhoods, normals & tangents"""
    _print('building point neighborhoods')
    _print('setting up initial tree', 1)
    tree = BallTree(points, leaf_size=64)
    _print('querying neighbors', 1)
    max_neighbors = 2 * k
    neighbors = tree.query(
        points, max_neighbors, sort_results=True, dualtree=True,
        return_distance=False)
    if normals is None:
        _print('computing normals', 1)
        normals = compute_normals(points, neighbors)
    _print('computing tangents', 1)
    tangents = compute_tangent_planes(points, neighbors, normals)
    _print('(a_neighbors computation disabled)', 1)
    _print('neighborhoods done')
    return (
        neighbors[:, 1:k + 1], normals, tangents)


# deprecated
def circumradius(points, simplices):
    """
    compute circumradius of input simplices
    (see https://plotly.com/python/v3/alpha-shapes/)
    """
    A = points[simplices]
    M = np.zeros((A.shape[0], 3, 4))
    M[:, :, -1] = 1
    M[:, :, 0] = np.sum((A ** 2), axis=(-1))
    M[:, :, 1] = A[:, :, 0]
    M[:, :, 2] = A[:, :, 1]
    S = np.column_stack([0.5 * np.linalg.det(M[:, :, [0, 2, 3]]),
                         -0.5 * np.linalg.det(M[:, :, [0, 1, 3]])])
    a = np.linalg.det(M[:, :, 1:])
    b = np.linalg.det(M[:, :, [0, 1, 2]])
    i = abs(a) > 1e-10
    r = np.zeros(simplices.shape[0])
    r[i] = np.sqrt(b[i] / a[i] + (S[(i, 0)] ** 2 + S[(i, 1)] ** 2) / a[i] ** 2)
    r[~i] = np.inf
    return r


# deprecated
def alpha_shape_removed(triangulation, alpha):
    """
    indices of the simplices which are not part of the corresponding alpha shape
    """
    i = circumradius(triangulation.points, triangulation.simplices) > alpha
    return i


# def process_point_neighborhood(i, h, points, neighbors, normals, tangents,
#                                k_neighbors, a_neighbors, inner, max_r, max_z, alpha=None):
#     """
#     deprecated.
#     for point i, compute inner points & immediate neighbors
#     """
#     if alpha is None:
#         alpha = 1.0 / h
#     t1, t2 = tangents
#     px = points[i]
#     pns = points[neighbors[i]]
#     x = (pns - px) @ t1[i]
#     y = (pns - px) @ t2[i]
#     z = (pns - px) @ normals[i]
#     candidates = [j for j in np.argsort(x ** 2 + y ** 2) if abs(z[j]) < max_z]
#     k_neighbors[i] = [neighbors[i][j] for j in candidates[1:k_neighbors.shape[1] + 1]]
#     s = (np.sqrt(x ** 2 + y ** 2) < max_r) & (abs(z) < max_z)
#     pts = np.column_stack((x[s], y[s]))
#     nbs = np.r_[neighbors[i]][s]
#     inner_ = np.ones((pts.shape[0]), dtype=bool)
#     tri = Delaunay(pts)
#     indptr, indices = tri.vertex_neighbor_vertices
#     ch = np.unique(tri.convex_hull)
#     outer = alpha_shape_removed(tri, alpha=alpha).flatten()
#     ah = np.unique(tri.simplices[outer])
#     edges = np.unique(np.r_[(ch, ah)])
#     inner_[edges] = False
#     for m in edges:
#         inner_[indices[indptr[m]:indptr[(m + 1)]]] = False
#         neighbor_simplices = np.any((tri.simplices == 0), axis=(-1))
#         new = np.unique([nbs[k] for k in tri.simplices[(neighbor_simplices & ~outer)]])
#         a_neighbors[i].update(new)
#         for j in new:
#             a_neighbors[j].add(i)
#         else:
#             inner[nbs[inner_]] = True


class Derivative:
    def __init__(self, points, max_d, neighbors, tangents):
        """precalculate inverse for WLS approximation via SVD"""
        n = points.shape[0]
        t1, t2 = tangents
        M = np.zeros((n, neighbors.shape[1]+3, 5))
        x_ = np.sum(
            ((points[neighbors, :] - points[:, None, :]) * t1[:, None, :]), axis=(-1))
        y_ = np.sum(
            ((points[neighbors, :] - points[:, None, :]) * t2[:, None, :]), axis=(-1))
        self.x, self.y = x_, y_
        M[:, :-3, 0] = x_
        M[:, :-3, 1] = y_
        M[:, :-3, 2] = x_ * y_
        M[:, :-3, 3] = x_ ** 2
        M[:, :-3, 4] = y_ ** 2
        M[:, -3, 2] = 1
        M[:, -2, 3] = 1
        M[:, -1, 4] = 1
        self.weights = (np.sqrt(wendland(np.linalg.norm(
            points[neighbors, :] - points[:, None, :], axis=-1),
            h=max_d)) + 1e-8) / (1 + 1e-8)
        M[:, :-3, :] *= self.weights[:, :, None]
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        if np.any(np.abs(S) < 1e-12):
            import pdb
            pdb.set_trace()
        self.D = np.einsum('nij,nki->njk', Vh, 1 / S[:, None, :] * U)
        self.neighbors = neighbors
        self.points = points
        self.t1, self.t2 = tangents

    def __call__(self, u, idx=None, first_only=False):
        """first & optionally second derivatives of u at idx w.r.t. local tangent basis"""
        if idx is None:
            idx = slice(None)
        f = np.zeros((self.weights[idx].shape[0], self.neighbors.shape[1] + 3))
        f[:, :-3] = self.weights[idx] * \
            (u[self.neighbors[idx]] - u[(idx, None)])
        f[:, -3:] = 1.0 / 1000
        if first_only:
            a = np.einsum('nij, nj->ni', self.D[idx][:, :2, :], f)
            return (
                a[:, 0], a[:, 1])
        a = np.einsum('nij, nj->ni', self.D[idx], f)
        return (
            a[:, 0], a[:, 1], 2 * a[:, 3], 2 * a[:, 4])

    def grad(self, u, idx):
        """gradient of u at idx"""
        if idx is None:
            idx = slice(None)
        dx, dy = self(u, idx, first_only=True)
        return dx[:, None] * self.t1[idx] + dy[:, None] * self.t2[idx]

    def div(self, u, idx):
        """divergence of u at idx"""
        dx_ux = self.grad(u[:, 0], idx)[:, 0]
        dy_uy = self.grad(u[:, 1], idx)[:, 1]
        dz_uz = self.grad(u[:, 2], idx)[:, 2]
        return dx_ux + dy_uy + dz_uz


# def heaviside(x, epsilon=1.0):
#     """mollified heaviside function"""
#     h = np.zeros_like(x)
#     i1 = abs(x) <= epsilon
#     i2 = x > epsilon
#     h[i1] = 0.5 * (1 + x[i1] / epsilon + 1 / np.pi *
#                    np.sin(np.pi * x[i1] / epsilon))
#     h[i2] = 1
#     return h


def heaviside(x, epsilon=1.0):
    """mollified heaviside function"""
    return 0.5 * (1 + 2 / np.pi * np.arctan(x / epsilon))


# def delta(x, epsilon=1.0):
#     """mollified dirac delta"""
#     d = np.zeros_like(x)
#     i1 = abs(x) <= epsilon
#     d[i1] = 1 / (2 * epsilon) * (1 + np.cos(np.pi * x[i1] / epsilon))
#     return d


def delta(x, epsilon=1.0):
    return 1/np.pi * epsilon / (epsilon**2 + x**2)


def dp(s):
    """dp(s)=d'(s)/s where d is the double-well potential"""
    st1 = s < 1
    dp_ = np.zeros_like(s)
    dp_[st1] = np.sinc(2 * s[st1])
    dp_[~st1] = 1 - 1 / s[(~st1)]
    return dp_


def g(a):
    """edge detector"""
    return 1 / (1 + np.linalg.norm((grad(a, None)), axis=(-1)) ** 2)


def smooth(u, idx, nbs, w):
    """kernel smoother with weights w"""
    shape = u[idx].shape
    if len(u.shape) <= 1:
        u = np.expand_dims(u, -1)
    s = (u[nbs[idx]] * np.expand_dims(w[idx], -1)
            ).sum(axis=(-2)) / w[idx].sum(axis=(-1))[:, None]
    s = s.reshape(shape)
    return s


class Solver:
    def __init__(self, h, zeta, points, neighbors, tangents,
                 active_contour_model='chan_vese'):
        """level set solver"""
        _print('initializing solver')
        self.phi = np.zeros(zeta.shape[:-1])
        self.zeta = zeta.copy()
        self.h = h
        self.points = points
        self.neighbors = neighbors
        self.max_val = 2 * h
        self.active = np.zeros((self.phi.shape), dtype='bool')
        self.last_active = self.active.copy()
        self.laplace = np.zeros_like(self.phi)
        self.dphi = np.zeros((self.phi.shape[0], 3))
        self.absdphi = np.zeros_like(self.phi)
        self.dp_ = np.zeros_like(self.phi)
        self.ctmp = np.zeros_like(self.dphi)
        self.seeds = None
        self.active_contour_model = active_contour_model
        _print('initializing WLS derivatives', 1)
        self.diff = Derivative(points, 3 * h, neighbors, tangents)
        self.neighborhood = np.column_stack(
            (np.arange(self.points.shape[0]), self.neighbors))
        
    def _chan_vese_step(self, idx, epsilon, nu, mu, lambda1, lambda2):
        if idx is None:
            idx = slice(None)

        R = 0
        C = 0
        F = 0

        if nu > 0 or mu > 0:
            dx, dy, ddx, ddy = self.diff(self.phi, idx)
            self.laplace[idx] = ddx + ddy
            self.dphi[idx] = (dx[:, None] * self.diff.t1[idx] +
                              dy[:, None] * self.diff.t2[idx])
            self.absdphi[idx] = np.linalg.norm((self.dphi[idx, :]), axis=(-1))
            idx = idx.copy() * (self.absdphi > 0)
            self.dp_[~idx] = 0
            self.dp_[idx] = dp(self.absdphi[idx])

        if nu > 0:
            # distance regularization
            R = nu * (self.dp_[idx] * self.laplace[idx] +
                      np.sum((self.diff.grad(self.dp_, idx) *
                              self.dphi[idx]), axis=(-1)))

        Hphi = heaviside(self.phi, epsilon)[:, None]
        zeta_in = (self.zeta * (1 - Hphi)).sum(axis=0) / (1 - Hphi).sum()
        zeta_out = (self.zeta * Hphi).sum(axis=0) / Hphi.sum()

        if mu > 0:
            self.ctmp[:] = 0
            self.ctmp[idx, :] = self.dphi[idx] / self.absdphi[(idx, None)]
            curvature = self.diff.div(self.ctmp, idx)
            C = mu * curvature

        #  if lambda1 is None: 
        #      sigma_in = ( 
        #          ((self.zeta[idx] - zeta_out)**2 * 
        #           (1 - Hphi[idx])).sum(axis=-1) / 
        #          (1 - Hphi[idx]).sum(axis=-1))**0.5 
        #      lambda1 = 1/sigma_in 
        #  if lambda2 is None: 
        #      sigma_out = ( 
        #          ((self.zeta[idx] - zeta_in)**2 * 
        #           (Hphi[idx])).sum(axis=-1) / 
        #          (Hphi[idx]).sum(axis=-1))**0.5 
        #      lambda2 = 1/sigma_out 

        D = delta(self.phi[idx], epsilon)
        F = (lambda1 * np.sum((self.zeta[idx] - zeta_in) ** 2, axis=-1)
             - lambda2 * np.sum((self.zeta[idx] - zeta_out) ** 2, axis=-1))

        return (idx, R + D * (C + F))

    def _lmv_step(self, idx, epsilon, mu, nu):
        dx, dy, ddx, ddy = self.diff(self.phi, idx)
        self.laplace[idx] = ddx + ddy
        self.dphi[idx] = (dx[:, None] * self.diff.t1[idx] +
                          dy[:, None] * self.diff.t2[idx])
        self.absdphi[idx] = np.linalg.norm((self.dphi[idx, :]), axis=(-1))
        idx = idx.copy() * (self.absdphi > 0)
        self.dp_[~idx] = 0
        self.dp_[idx] = dp(self.absdphi[idx])

        # distance regularization
        R = nu * (self.dp_[idx] * self.laplace[idx] +
                  np.sum((self.diff.grad(self.dp_, idx) *
                          self.dphi[idx]), axis=(-1)))
        self.ctmp[:] = 0
        self.ctmp[idx, :] = self.dphi[idx] / self.absdphi[(idx, None)]
        curvature = self.diff.div(self.ctmp, idx)
        C = mu * curvature

        Hphi = heaviside(self.phi, epsilon)
        nb = self.neighborhood
        m = self.neighborhood.shape[1]

        idx_ = idx.copy()
        idx_[nb[idx]] = True

        mu_out = np.zeros_like(self.zeta)
        mu_in = np.zeros_like(self.zeta)
        sigma_out = np.zeros_like(self.zeta)
        sigma_in = np.zeros_like(self.zeta)

        nb_ = nb[idx_]

        mu_in[idx_] = (
            (self.zeta[nb_] * (1 - Hphi[nb_])).sum(axis=-1) /
            (1 - Hphi[nb_]).sum(axis=-1))
        mu_out[idx_] = (
            (self.zeta[nb_] * Hphi[nb_]).sum(axis=-1) /
            (Hphi[nb_]).sum(axis=-1))

        sigma_in[idx_] = (
            ((self.zeta[nb_] - mu_in[idx_, None])**2 *
             (1 - Hphi[nb_])).sum(axis=-1) /
            (1 - Hphi[nb_]).sum(axis=-1))**0.5
        sigma_out[idx_] = (
            ((self.zeta[nb_] - mu_out[idx_, None])**2 *
             (Hphi[nb_])).sum(axis=-1) /
            (Hphi[nb_]).sum(axis=-1))**0.5

        i = idx  # & (sigma_in > 0) & (sigma_out > 0)
        i_ = i[idx]

        sigma_in[idx_] = np.maximum(sigma_in[idx_], 0.0001)
        sigma_out[idx_] = np.maximum(sigma_out[idx_], 0.0001)

        k = idx.sum()
        e_in = np.zeros(k)
        e_out = np.zeros(k)

        e_in[i_] = 1/m * (
            np.log(np.sqrt(2*np.pi) * sigma_out[i, None]) +
            (self.zeta[nb[i]] - mu_out[nb[i]])**2 /
            (2*sigma_out[i, None]**2)
        ).sum(axis=-1)
        e_out[i_] = 1/m * (
            np.log(np.sqrt(2*np.pi) * sigma_in[i, None]) +
            (self.zeta[nb[i]] - mu_in[nb[i]])**2 /
            (2*sigma_in[i, None]**2)
        ).sum(axis=-1)

        F = e_in - e_out

        D = delta(self.phi[idx], epsilon)

        return idx, D * (C + F) + R

    def _step(self, idx, epsilon, **kwargs):
        """one explicit euler step (unscaled)"""
        assert self.active_contour_model in ['chan_vese', 'lmv']
        if self.active_contour_model == 'chan_vese':
            return self._chan_vese_step(idx, epsilon, **kwargs)
        else:
            return self._lmv_step(idx, epsilon,
                                  mu=kwargs['mu'], nu=kwargs['nu'])

    def zero_crossings(self, u, idx=None, neighbors=None):
        """
        get all points (indices) which have a neighbor with differing sign
        """
        if neighbors is None:
            neighbors = self.neighbors
        if idx is None:
            idx = slice(None)
        if type(neighbors) is np.ndarray:
            d = np.sign(u[idx][:, None]) != np.sign(u[neighbors[idx]])
            return np.any(d, axis=(-1))
        zc = np.zeros((u[idx].size), dtype=bool)
        for i in range(zc.size):
            nbs = np.array(neighbors[idx][i])
            zc[i] = np.any(np.sign(u[idx][i]) != np.sign(u[nbs]))
        else:
            return zc

    def initialize_new(self, new, active):
        self.phi[new] = np.sign(self.phi[new]) * self.max_val
        self.phi[self.neighbors[(new & ~active)]] = np.sign(
            self.phi[(new & ~active), None]) * self.max_val

    def initialize_from_mask(self, mask):
        _print('initializing from mask')
        _print('setting phi to +=max_val', 1)
        self.phi[mask] = -self.max_val
        self.phi[~mask] = self.max_val
        self.active = self.zero_crossings(self.phi)
        self.last_active[:] = self.active

    def initialize_from_zeta(self, percentile):
        _print('initializing from zeta')
        _print('setting phi to +=max_val', 1)
        pc = np.percentile(self.zeta, percentile)
        self.phi[self.zeta <= pc] = -self.max_val
        self.phi[self.zeta > pc] = self.max_val
        self.active = self.zero_crossings(self.phi)
        self.last_active[:] = self.active
        _print('setting zeta=0 on boundary', 1)

    def initialize_from_seeds(self, num_seeds, seed_radius, reuse_seeds=True):
        if self.seeds is None or not reuse_seeds:
            _print('generating seeds')
            self.seeds = np.zeros((self.points.shape[0]), dtype=bool)
            for _ in range(num_seeds):
                i = np.random.choice(self.points.shape[0], 1)[0]
                self.seeds[np.linalg.norm(
                    (self.points - self.points[i]), axis=1) < seed_radius
                ] = True

        else:
            _print('reusing seeds')
        _print('initializing from seeds')
        self.phi[self.seeds] = self.max_val
        self.phi[~self.seeds] = -self.max_val
        self.active = self.zero_crossings(self.phi)
        self.last_active = self.active

    def initialize_from_point(self, point, radius):
        i = np.linalg.norm((point - self.points), axis=(-1)) <= radius
        _print('initializing from radius')
        self.phi[~i] = self.max_val
        self.phi[i] = -self.max_val
        self.active = self.zero_crossings(self.phi)
        self.last_active = self.active

    def initialize_from_voxels(self, h):
        i = np.sum((self.points // h), axis=(-1)) % 2 == 0
        self.phi[~i] = self.max_val
        self.phi[i] = -self.max_val
        self.active = self.zero_crossings(self.phi)
        self.last_active = self.active

    def run(self, iterations, stepsize,
            nu, mu, lambda1, lambda2, epsilon, cap=True, tolerance=1e-4):
        _print(f"running {iterations} steps")
        new = np.zeros_like(self.active)
        for i in range(iterations):
            new[self.active] = True
            new[self.last_active] = False
            self.phi[~self.active & self.last_active] = \
                np.sign(
                    self.phi[(~self.active & self.last_active)]) * self.max_val
            self.initialize_new(new, self.active)
            active, dphi = self._step(
                self.active,
                epsilon=epsilon * self.h, nu=nu, mu=mu,
                lambda1=lambda1, lambda2=lambda2,
            )
            increment = stepsize * dphi
            self.phi[active] += increment
            rmsc = np.mean(increment**2)**0.5 # mean squared change
            self.last_active[:] = self.active
            self.active[self.active] = self.zero_crossings(
                self.phi, self.active)
            if (mu > 0 or nu > 0) and cap:
                outside_limits = (self.phi > self.max_val) | (
                    self.phi < -self.max_val)
                self.phi[outside_limits] = \
                    np.sign(self.phi[outside_limits]) * self.max_val
                self.phi[outside_limits] = \
                    smooth(self.phi, outside_limits,
                           self.neighbors, self.diff.weights)
            lonely = (np.all((np.sign(self.phi[self.neighbors])
                              != np.sign(self.phi)[:, None]), axis=(-1)))
            self.phi[lonely] = smooth(
                self.phi, lonely, self.neighbors, self.diff.weights)
            self.active[self.neighbors[self.active]] = True
            _print(f"\r{i + 1}/{iterations} \t RMSE(dphi): {rmsc:.3e}",
                   1, end='')
            if rmsc < tolerance:
                _print(f"\nnumber of active points: {self.active.sum()}", 1)
                _print("tolerance reached!")
                return True

        _print("\r")
        _print(f"\nnumber of active points: {self.active.sum()}", 1)
        return False

    def save(self, basepath, origin=np.r_[0, 0, 0],
             extract=False, extraction_threshold=4):
        out = np.column_stack((self.points[self.active]+origin,
                               self.phi[self.active],
                               self.zeta[self.active]))
        np.savetxt(basepath+'.txt', out, fmt='%1.6f')

        if extract:
            region1 = (self.phi > 0)
            region2 = (self.phi <= 0)
            if (self.zeta[region1].mean() > self.zeta[region2].mean()):
                salient = region1
            else:
                salient = region2

            salient |= (np.sum(salient[self.neighbors], axis=-1) >=
                        extraction_threshold)
            salient &= np.any(salient[self.neighbors], axis=-1)
            out = np.column_stack(
                (self.points[salient]+origin[None, :],
                 self.phi[salient],
                 self.zeta[salient])
            )
            np.savetxt(basepath+'_extract.txt', out, fmt='%1.6f')

#     def level_set_points(self, neighbors=None):
#         if neighbors is None:
#             neighbors = self.neighbors
#         _print('generating points that lie on the zero-level set [phi=0]')
#         phi = self.phi
#         zc = self.zero_crossings(phi, neighbors=neighbors)
#         zc_ = []
#         for i in np.where(zc)[0]:
#             u = phi[i]
#             x = self.points[i]
#             nbs = np.array((neighbors[i]), dtype=int)
#             if not np.any((np.sign(phi[nbs]) == np.sign(u)) & (abs(phi[nbs]) < abs(u)) & (np.linalg.norm((x - self.points[nbs]), axis=(-1)) < 0.5 * self.h)):
#                 zc_.append(i)
#         else:
#             pts = []
#             for i in zc_:
#                 u1 = phi[i]
#                 if u1 < 0:
#                     pass
#                 else:
#                     nbs_ = np.array([j for j in neighbors[i] if phi[j] < 0])
#                     nbs_ = nbs_[np.argsort(phi[nbs_])[-2:]]
#                 for j in nbs_:
#                     u2 = phi[j]
#                     x1 = self.points[i]
#                     x2 = self.points[j]
#                     t = -u1 / (u2 - u1)
#                     x_ = x1 + t * (x2 - x1)
#                     pts.append(x_)

#             else:
#                 pts = np.array(pts)
#                 return pts
