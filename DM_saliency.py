'''
===================================================================
                            MorE3d
===================================================================

Functionaliries to compute saliency based on geometric properties
-----------------------------------------------------------------
@author: Reuma Arav
@date: 2022

# Dependencies

    - OPALS v.2.5.0

When using this code please cite:

@article{arav2022visual,
  title={A visual saliency-driven extraction framework of smoothly embedded entities in 3D point clouds of open terrain},
  author={Arav, Reuma and Filin, Sagi},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={188},
  pages={125--140},
  year={2022},
  publisher={Elsevier}
}

'''

from __future__ import print_function  # print function syntax as in python 3

import warnings
import opals.AddInfo
from opals import pyDM, Info, Histo


import datetime
import numpy as np
import sys

class Curvature_Kernel(pyDM.PointKernelEx):
    """callback object computing an user-defined attribute '_K_nonparam'"""
    _roughness = None
    _alpha = None
    _minPoints_neighbourhood = None

    def __init__(self):
        """
        initialize the kernel process, and set constant parameters

        """

        # initialize the base class (mandatory!)
        if sys.version_info >= (3, 0):
            super().__init__()
        else:
            super(Curvature_Kernel, self).__init__()
        return

    def setArgs(self, alpha, roughness, min_points = 4):
        """
        Sets the class private arguments roughness and alpha

        :param roughness: minimal object size - usually refers to surface roughness
        :param alpha: confidence value for statistical test
        :param min_points: minimum points in the neighbourhood. Default: 4

        :type alpha: float
        :type roughness: float
        :type min_points: int

        """
        self._alpha = alpha
        self._roughness = roughness
        self._minPoints_neighbourhood = min_points

    def tileFilter(self):
        return None

    def leafChanged(self, leaf, *args, **kwargs):
        """callback notifies about a changed leaf"""
        # print("process tile with id = %d" % leaf.id())
        return

    def process(self, pt, neighbours):
        r"""
        callback for computing a point's non parametric curvature

        .. math::
            _nonparam_K= \frac{1}{k}\sum_{i=1}^k {\bf n}_q}^T {\bf p_i}
        """
        from scipy import stats
        # neighbours = args[1]
        # pt = args[0]
        # print(pt.info().get(5))
        epsilon = stats.norm.ppf(1 - self._alpha / 2) * self._roughness  # Z-distribution

        # set the number of neighbours in each point
        pt.info().set(4, neighbours.sizePoint())
        proj = []
        proj_tangent = []

        # check that the point info is full
        if pt.info().isNull(0) == True:
            warnings.warn('Point does not have normal information, set 0')
            pt.info().set(3, 0)
            return True

        normal_pt = np.array([pt.info().get(0),
                              pt.info().get(1),
                              pt.info().get(2)])

        # projection matrix for non-viable points
        P_q = np.eye(3) - np.outer(normal_pt, normal_pt)
        pt_projected = P_q.dot(np.array([pt.x, pt.y, pt.z]))

        if neighbours.sizePoint() < self._minPoints_neighbourhood:
            print('point does not have enough neighbours, curvature set to zero')
            pt.info().set(3, 0)
            return True

        neighbours_dict = neighbours.asNumpyDict()
        n_xyz = np.vstack((neighbours_dict['x'], neighbours_dict['y'], neighbours_dict['z']))
        pt_xyz = np.hstack((pt.x, pt.y, pt.z))

        # pt_xyz = np.hstack((pt.x, pt.y, pt.z))

        if normal_pt.dot(-pt_xyz) < 0:
            normal_pt = -normal_pt
            pt.info().set(0, normal_pt[0])
            pt.info().set(1, normal_pt[1])
            pt.info().set(2, normal_pt[2])

        dxyz = pt_xyz - n_xyz.T
        proj_np = normal_pt.dot(dxyz.T)
        proj_tangent = P_q.dot(n_xyz)

        cg = np.mean(proj_tangent.T - pt_projected, axis=0)
        if np.linalg.norm(cg) > 1:
            pt.info().set(3, 0)
            return True

        sum_proj = np.sum(proj_np) / (neighbours.sizePoint() - 1)

        # check if the projections are statistically zero
        if np.abs(sum_proj) < epsilon:
            pt.info().set(3, 0)
        else:
            # print(sum_proj)
            pt.info().set(3, sum_proj)  # else, it is the mean of the projections

        # return True if the point was changed otherwise False
        # this internally marks the current point leaf as changed (=needs to be written to disk)
        return True

class Saliency_Kernel(pyDM.PointKernelEx):
    """callback object computing an user-defined attribute '_saliency'"""
    _alpha = None
    _sigma_curvature = None
    _sigma_normal = None
    _normal_weight = None
    _sigma_neighbourhood = None
    _rho_neighbourhood = None
    _minPoints_neighbourhood = None

    def __init__(self):
        """
        initialize the kernel process, and set constant parameters

        """

        # initialize the base class (mandatory!)
        if sys.version_info >= (3, 0):
            super().__init__()
        else:
            super(Saliency_Kernel, self).__init__()
        return

    def setArgs(self, rho_neighbourhood, sigma_neighbourhood,  sigma_normal=0.01, sigma_curvature= 0.1,
                alpha=0.05, min_points=4):
        """
        Sets the class private arguments for saliency computation

        :param rho_neighbourhood: the effective distance (for weighting function)
        :param sigma_neighbourhood: the width of the gaussian (for weighting function)
        :param sigma_normal: maximal std of the normals deviations for a point to be considered as vegetation. Default: 0.01
        :param sigma_curvature: std of the curvature deviations for a surface texture. Default: 0.1
        :param alpha: for the hypothesis testing. Default: 0.05
        :param min_points: minimum points in the neighbourhood. Default: 4

        :type sigma_normal: float
        :type sigma_curvature: float
        :type alpha: float
        :type min_points: int
        """
        self._alpha = alpha
        self._sigma_curvature = sigma_curvature
        self._sigma_normal = sigma_normal
        self._sigma_neighbourhood = sigma_neighbourhood
        self._rho_neighbourhood = rho_neighbourhood
        self._minPoints_neighbourhood = min_points

    def tileFilter(self):
        return None

    def leafChanged(self, leaf, *args, **kwargs):
        """callback notifies about a changed leaf"""
        # print("process tile with id = %d" % leaf.id())
        return

    # @jit(nopython=True)
    def process(self, pt, neighbours):
        r"""
        callback for computing a point's saliency
        """
        from scipy import stats
        # check that the point info is full
        if pt.info().isNull(0):
            print("No normal information, dn and dk set to zero")
            pt.info().set(4, 0)
            pt.info().set(5, 0)
            return True

        pt_normal = np.array([pt.info().get(0),
                              pt.info().get(1),
                              pt.info().get(2)])
        pt_curvature = pt.info().get(3)
        pt_xyz = np.hstack((pt.x, pt.y, pt.z))

        # if pt_normal.dot(-pt_xyz) < 0:
        #     pt_normal = -pt_normal

        if neighbours.sizePoint() < self._minPoints_neighbourhood:
            print('point does not have enough neighbours, dk and dn set to zero')
            pt.info().set(4, 0)
            pt.info().set(5, 0)
            return True

        neighbours_dict = neighbours.asNumpyDict()
        n_xyz, idx = np.unique(np.vstack((neighbours_dict['x'], neighbours_dict['y'], neighbours_dict['z'])), axis=1,
                          return_index=True)
        n_curvature = neighbours_dict['_nonparam_K'][idx]

        n_normal = np.vstack((neighbours_dict['NormalX'], neighbours_dict['NormalY'], neighbours_dict['NormalZ']))[:, idx]
        # turn all normals to the same direction (of [0,0,0]
        # ind = np.nonzero(np.einsum('ij,ij->i', n_normal.T, -n_xyz.T) < 0)
        # n_normal.T[ind, :] = -n_normal.T[ind,:]

        dxyz = pt_xyz - n_xyz.T
        dist = np.sum(dxyz**2, axis=1)
        dn = 1- pt_normal.dot(n_normal)
        dk = pt_curvature - n_curvature

        # compute the weight of the neighbour using the Gaussian weighting function
        dist = np.asarray(dist)
        neighbourhood_weights = np.zeros(dist.shape[0])

        # neighbourhood_weights = 1 / np.sqrt(2 * np.pi * self._sigma_neighbourhood ** 2) * \
        #            np.exp(-(dist - self._rho_neighbourhood) ** 2 / (2 * self._sigma_neighbourhood ** 2))

        # approximate weighting function to triangle with base 2*rho**2
        rho2 = self._rho_neighbourhood**2
        neighbourhood_weights[np.abs(dist) < rho2] = 1/rho2 * np.abs(dist[np.abs(dist) < rho2])
        neighbourhood_weights[np.abs(dist) > rho2] = -1/rho2 * np.abs(dist[np.abs(dist) > rho2]) + 2
        neighbourhood_weights[neighbourhood_weights < 0] =0


        # statistical tests for zero value
        active_w = neighbourhood_weights > 0.01
        number_active = int(
            np.sum(active_w))  # how many of the neighbors take part in the computation
        chi2_table = stats.chi2.ppf(self._alpha, number_active)  # X^2-distribution
        chi2_normal = (neighbours.sizePoint() - 1) * dn.std() / self._sigma_normal
        chi2_curvature = (neighbours.sizePoint() - 1) * dk.std() / self._sigma_curvature

        num_zeros_normals = np.sum(np.abs(dn[active_w]) < 0.016)
        num_zeros_curvature = np.sum(np.abs(dk[active_w]) <= 0.04)

        if num_zeros_curvature == number_active:
            num_zeros_curvature = np.sum(np.abs(dk[active_w] * 100) <= 0.04)

        if chi2_normal < chi2_table:
            # if most normal differences are statistically zero
            dn = 0

        elif num_zeros_normals > 0.6*number_active:
            # if there are more than 60% that are less than 5 deg difference
            dn =0
        else:
            dn = np.sum(np.abs(dn) * neighbourhood_weights)/np.sum(neighbourhood_weights)
            # if dn > 1:
            #     print('hello')
        if np.isnan(dn):
            dn = 0
            print('dn is nan')
        elif dn > 2:
            dn = 1
            print('dn is larger than 2')
        if chi2_curvature < chi2_table:
            # if most curvature differences are statistically zero
            dk = 0
        elif num_zeros_curvature > 0.6 * number_active:
            # if there are more than 60% that are less than 0.04 deg difference
            dk = 0
        else:
            dk = np.abs(np.sum(dk * neighbourhood_weights)/np.sum(neighbourhood_weights))
        # print('dn {}, dk {}'.format(dn, dk))

        pt.info().set(4, dk)
        pt.info().set(5, dn)

        # return True if the point was changed otherwise False
        # this internally marks the current point leaf as changed (=needs to be written to disk)
        return True

def DM_nonparam_curvature(inFile, shape, roughness=0.15, alpha=0.05, min_points = 3, **kwargs):
    r"""
    Compute the non parametric curvature for each point based on a neighbourhood that is defined by a shape and its attirbutes.
    .. note::
         The odm should be after the normals computation

    .. math:
        \kappa= \frac{1}{k}\sum_{i=1}^k {\bf n}_q}^T {\bf p_i}

    with :math:`{\bf p_i}` the vector from one neighbour to the feature point, and :math:`{\bf n}_q}` the normal at the feature point

    :param inFile: path to the odm file
    :param shape: the shape in which the neighbours will be searched in. Options:
        - 'circle'
        - 'sphere'
        - 'cylinder'
    :param roughness: minimal object size - usually refers to surface roughness
    :param alpha: confidence value for statistical test
    :param min_points: minimum points in neighbourhood
    :param kwargs: attributes of the shape: For:
        - circle: 'radius'
        - sphere: 'radius'
        - cylinder: 'radius', 'zmin', 'zmax'

    :type inFile: str
    :type shape: str
    :type alpha: float
    :type roughness: float
    :type min_points: int
    :type radius: float
    :type zmin: float
    :type zmax: float

    :return: a list of each point's neighbours
    """
    try:
        dm = pyDM.Datamanager.load(inFile, False, False)
    except IOError as e:
        print(e)
        return

    # sanity check
    if 'radius' not in kwargs:
        print("must have radius for search")
        sys.exit(1)
    if shape == 'cylinder':
        if 'zmin' not in kwargs or 'zmax' not in kwargs:
            print("for cylinder zmin and zmax must be defined for search")
            sys.exit(1)

    # initialize an empty layout
    lf = pyDM.AddInfoLayoutFactory()
    lf.addColumn(pyDM.ColumnSemantic.NormalX)  # column 0
    lf.addColumn(pyDM.ColumnSemantic.NormalY)  # column 1
    lf.addColumn(pyDM.ColumnSemantic.NormalZ)  # column 2
    lf.addColumn(pyDM.ColumnType.float_, "_nonparam_K")  # column 3
    lf.addColumn(pyDM.ColumnType.int32, "_pcount")  # column 4
    # lf.addColumn(pyDM.ColumnSemantic.Id) # column 5
    layout = lf.getLayout()

    # create spatial selection
    # define the shape in which the neighbourhood will be searched
    if shape == 'circle':
        # TODO: check if the circle query works.
        search_shape = pyDM.QueryCircle(r=kwargs['radius'])
    elif shape == 'sphere':
        # search_shape = pyDM.QuerySphere(r=kwargs['radius'])
        search_shape = 'sphere(r=' + str(kwargs['radius']) + ')'
    elif shape == 'cylinder':
        # TODO: check if the cylinder query works.
        search_shape = pyDM.QueryCylinder(r=kwargs['radius'], zmin=kwargs['zmin'], zmax=kwargs['zmax'])

    query = pyDM.QueryDescriptor(search_shape)  # creates the query according to the shape defined
    k = Curvature_Kernel()
    k.setArgs(alpha=alpha, roughness=roughness, min_points=min_points)
    start = datetime.datetime.now()
    # create processor according to which the kernel will run (sends a point and its neighbours to the kernel)
    processor = pyDM.ProcessorEx(dm, query, None, layout, False, None, None, False)
    print("Computing non-parametric curvature... ")
    print('Process started at', datetime.datetime.now().strftime('%H:%M:%S'))

    processor.run(k)  # perform computation

    diff = datetime.datetime.now() - start

    print("Done. Non-parametric curvature computation took %.2f [s]" % diff.total_seconds())

    # save manager object
    print("Save...")
    dm.save()

    return dm

def DM_dk_dn(inFile, shape,  rho_neighbourhood, sigma_neighbourhood, sigma_normal=0.01, sigma_curvature=0.1,
                alpha=0.05, min_points=4, **kwargs):
    r"""
    Compute diffrences of curvature and normals (as a preparation for saliency)

    .. note::
         The odm should be after normals and curvature computation

    .. math::
        d{\bf n}(\textbf{q}) = \frac{\iint_\mathcal{S}\left||{\n}(\textbf{q}) - {\n}(x,y)|\right| \cdot w(x,y) dxdy}
        {\iint_\mathcal{S} w(x,y)dxdy} \label{eq:saliency_normal}\\
        d\kappa(\textbf{q}) =  \frac{\iint_\mathcal{S}\left[\kappa(\textbf{q}) - \kappa(x,y)\right] \cdot w(x,y) dxdy}
        {\iint_\mathcal{S} w(x,y)dxdy}\label{eq:saliency_curvature}

    with:

    ..math::
        w(x,y) = \frac {1}{ \sqrt {2\pi } \sigma}\exp\left(-\frac{\left(\sqrt{x^2+y^2}-\rho\right)^2 }{2\sigma^2 }\right)

    :param inFile: path to the odm file
    :param shape: the shape in which the neighbours will be searched in. Options:
        - 'circle'
        - 'sphere'
        - 'cylinder'
    :param rho_neighbourhood: the effective distance (for weighting function)
    :param sigma_neighbourhood: the width of the gaussian (for weighting function)
    :param sigma_normal: maximal std of the normals deviations for a point to be considered as vegetation. Default: 0.01
    :param sigma_curvature: std of the curvature deviations for a surface texture. Default: 0.1
    :param min_points: minimum points in the neighbourhood. Default: 4
    :param kwargs: attributes of the shape: For:
        - circle: 'radius'
        - sphere: 'radius'
        - cylinder: 'radius', 'zmin', 'zmax'

    :type inFile: str
    :type shape: str
    :type sigma_normal: float
    :type sigma_curvature: float
    :type alpha: float
    :type min_points: int
    :type radius: float
    :type zmin: float
    :type zmax: float

    :return: a list of each point's neighbours
    """

    # import cProfile, pstats, io
    # pr = cProfile.Profile()
    # pr.enable()

    try:
        dm = pyDM.Datamanager.load(inFile, False, False)
    except IOError as e:
        print(e)
        return

    # sanity check
    if 'radius' not in kwargs:
        print("must have radius for search")
        sys.exit(1)
    if shape == 'cylinder':
        if 'zmin' not in kwargs or 'zmax' not in kwargs:
            print("for cylinder zmin and zmax must be defined for search")
            sys.exit(1)

    # initialize an empty layout
    lf = pyDM.AddInfoLayoutFactory()
    lf.addColumn(pyDM.ColumnSemantic.NormalX)  # column 0
    lf.addColumn(pyDM.ColumnSemantic.NormalY)  # column 1
    lf.addColumn(pyDM.ColumnSemantic.NormalZ)  # column 2
    lf.addColumn(pyDM.ColumnType.float_, "_nonparam_K")  # column 3
    lf.addColumn(pyDM.ColumnType.float_, "_dk")  # column 4
    lf.addColumn(pyDM.ColumnType.float_, "_dn")  # column 5
    lf.addColumn(pyDM.ColumnSemantic.Classification) #column 6
    layout = lf.getLayout()

    # create spatial selection
    # define the shape in which the neighbourhood will be searched
    if shape == 'circle':
        # TODO: check if the circle query works.
        search_shape = pyDM.QueryCircle(r=kwargs['radius'])
    elif shape == 'sphere':
        # search_shape = pyDM.QuerySphere(r=kwargs['radius'])
        search_shape = 'sphere(r=' + str(kwargs['radius']) + ')'
    elif shape == 'cylinder':
        # TODO: check if the cylinder query works.
        search_shape = pyDM.QueryCylinder(r=kwargs['radius'], zmin=kwargs['zmin'], zmax=kwargs['zmax'])

    query = pyDM.QueryDescriptor(search_shape)  # creates the query according to the shape defined
    s = Saliency_Kernel()
    s.setArgs(rho_neighbourhood, sigma_neighbourhood, sigma_normal, sigma_curvature, alpha, min_points)
    # create processor according to which the kernel will run (sends a point and its neighbours to the kernel)

    # pyDM.ProcessorEx parameter:  dm(datamnager), query(QueryDescriptor),
    #                              processFilter, processLayout, processLayoutReadOnly,
    #                              spatialFilter, spatialLayout, spatialLayoutReadOnly
    processor = pyDM.ProcessorEx(dm, query, None, layout, False, None, None, False)

    start = datetime.datetime.now()

    print("Computing dn and dk... ")
    print('Process started at', datetime.datetime.now().strftime('%H:%M:%S'))
    processor.run(s)  # perform computation

    diff = datetime.datetime.now() - start
    print("Done. dn and dk computation took %.2f [s]" % diff.total_seconds())
    print("Save...")
    dm.save()

    # pr.disable()
    # s = io.StringIO()
    # stats_out = 'profile.stats'
    # pr.dump_stats(stats_out)
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())


def DM_saliency(inFile, curvature_weight=.5, **kwargs):
    r"""
    Compute saliency of an odm.

     .. note::
         The odm should be after _dk and _dn computation

    .. math:
        S = w_{curvature} \cdot N(d\kappa) +w_{normal}\cdot N(d{\bf n})

    with :math:`N()` the normalized form of :math:`d{\bf n}` and :math:`d\kappa`.

    :param inFile: path to odm file
    :param curvature_weight: the weight of the curvature in the saliency computation

    :type inFile: str
    :type alpha: float

    :return: a list of each point's neighbours
    """
    start = datetime.datetime.now()
    print("Computing saliency... ")

    his = Histo.Histo(inFile=inFile, attribute=['_dk', '_dn'])
    his.run()

    min_k = his.histogram[0].getMin()
    max_k = his.histogram[0].getMax()
    min_n = his.histogram[1].getMin()
    max_n = his.histogram[1].getMax()

    diff_k = max_k - min_k
    diff_n = max_n - min_n
    normal_weight = 1 - curvature_weight
    print('diff_k {}, diff_n {}'.format(diff_k, diff_n))

    # AddInfo module
    print('Process started at', datetime.datetime.now().strftime('%H:%M:%S'))
    opals.AddInfo.AddInfo(inFile=inFile, attribute=f'_normed_dk(float) = (_dk - {min_k}) / {diff_k} + {min_k}').run() # normalize to values (0,1)
    opals.AddInfo.AddInfo(inFile=inFile, attribute=f'_normed_dn(float) = (_dn - {min_n}) / {diff_n} + {min_n}').run()  # normalize to values (0,1)
    opals.AddInfo.AddInfo(inFile=inFile, attribute=f'_saliency(float) = {normal_weight} * _normed_dn + {curvature_weight} '
                                        f'* _normed_dk').run()

    diff = datetime.datetime.now() - start
    print("Done. Processing took %.2f [s]" % diff.total_seconds())
