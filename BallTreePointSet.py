from functools import partial

import numpy as np
from numpy import nonzero, logical_and, unique, zeros, array, hstack
from sklearn.neighbors import BallTree

# from DataClasses.PointSet import PointSet


class BallTreePointSet(object):

    def __init__(self, points, leaf_size=40, **kwargs):
        r"""
        A ball tree representation

        :param points: the points to represent as ball tree
        :param leaf_size: minimal number of points in a leaf, with the maximal :math:`2\cdot` leaf_size

        :type points: PointSet, np.array
        :type leaf_size: int

          :type inputPoints: np.ndarray, o3D.PointCloud, PointSet

        """
        import open3d as O3D

        if isinstance(points, np.ndarray):
            self.data = BallTree(points, leaf_size)

        elif isinstance(points, O3D.PointCloud):
            pts = np.asarray(points.points)
            self.data = BallTree(pts, leaf_size)
        else:
            print("Given type: " + str(type(points)) + " as input. Points type should be numpy array or open3d PointCloud")
            raise ValueError("Wrong input object.")


        ballTreeData = self.data.get_arrays()  # Getting the data out of the ballTree object
        self.__ballTreePointIndexes = ballTreeData[1]  # Storing the internal indexes of the points
        self.__ballTreeNodes = ballTreeData[2]  # Storing the nodes data (mx4, ndarray)
        # self.__ballTreeNodeCenters = ballTreeData[3]  # Storing the centers of each node
        self.__numNodes = self.__ballTreeNodes.shape[0]  # Number of nodes in the tree
        self.__relations = []  # A list to define the tree hierarchy structure
        self.__ballTreeNodeHierarchy()  # Reconstructing the hierarchy of the tree
        self.__levels = zeros((self.__numNodes,), dtype=int)  # Array for storing the level of each node
        self.__ComputeNodeLevel(0)  # Computing the levels for all nodes
        # self.__numPnts = points.shape[0]

    # ---------------- PRIVATES-------------
    def __findChildNodes(self, i):
        """
        Finds the child nodes of the i-th node in the tree

        :param i: The index of the node to search for its child nodes

        :type i: int

        :return: A dictionary with the indexes of the left and right child nodes, if they exist

        :rtype: dict
        """
        childNodes = {'nodeId': i}

        if self.__ballTreeNodes['is_leaf'][i] == 0:  # Checking if node is not a leaf
            startIdx = self.__ballTreeNodes['idx_start'][i]  # The index of the first point in the node
            endIdx = self.__ballTreeNodes['idx_end'][i]  # The index of the last point in the node
            midIdx = int((startIdx + endIdx) / 2)  # The index of the middle point of the node

            # Finding the index of the left child of the node
            left = nonzero(logical_and(self.__ballTreeNodes['idx_start'][i + 1:] == startIdx,
                                       self.__ballTreeNodes['idx_end'][i + 1:] == midIdx))[0]

            # Finding the index of the right child of the node
            right = nonzero(logical_and(self.__ballTreeNodes['idx_start'][i + 1:] == midIdx,
                                        self.__ballTreeNodes['idx_end'][i + 1:] == endIdx))[0]

            if len(left) != 0:  # Checking if the left child node exists
                childNodes['leftChild'] = left[0] + i + 1  # Saving the index of left child node

            if len(right) != 0:  # Checking if the right child node exists
                childNodes['rightChild'] = right[0] + i + 1  # Saving the index of right child node

        return childNodes  # Return the dictionary with indexes of the child nodes

    def __setParent(self, i):
        """
        Sets ths i-th node as the parent of its child nodes

        :param i: The index of the node to set as parent

        :type i: int
        """
        left = False
        right = False

        if 'leftChild' in self.__relations[i]:  # Checking if the node has a left child
            left = True
            # Setting i as the parent of the left child node
            if not ('parent' in self.__relations[self.__relations[i]['leftChild']]):
                self.__relations[self.__relations[i]['leftChild']]['parent'] = i

        if 'rightChild' in self.__relations[i]:  # Checking if the node has a right child
            right = True
            # Setting i as the parent of the right child node
            if not ('parent' in self.__relations[self.__relations[i]['rightChild']]):
                self.__relations[self.__relations[i]['rightChild']]['parent'] = i

        if left and right:
            self.__relations[self.__relations[i]['leftChild']]['sibling'] = self.__relations[i]['rightChild']
            self.__relations[self.__relations[i]['rightChild']]['sibling'] = self.__relations[i]['leftChild']

    def __ballTreeNodeHierarchy(self):
        """
        Reconstruct the tree hierarchy based on the internal indexes of the points in each node

        """
        self.__relations = list(
            map(self.__findChildNodes, range(self.__numNodes)))  # Finding the children for each node
        list(map(self.__setParent, range(self.__numNodes)))  # Setting the parent for each node

    def __getFirstNodeOfSize(self, index, radius, startingNode='root'):
        """
        Get the first node along one of the tree branches whose radius is larger than a given one

        :param index: The index of the node to begin the search by
        :param radius: The minimal required size of the node

        :type index: int
        :type radius: float

        :return: The index of the first node with a radius larger than the given one

        :rtype: int
        """
        if startingNode == 'root':
            if self.__ballTreeNodes['radius'][index] < radius or self.__ballTreeNodes['is_leaf'][index] == 1:
                return index
            else:
                return hstack([self.__getFirstNodeOfSize(self.getLeftChildOfNode(index), radius, 'root'),
                               self.__getFirstNodeOfSize(self.getRightChildOfNode(index), radius, 'root')])

        elif startingNode == 'leaves':
            if self.__ballTreeNodes['radius'][index] < radius:

                while self.__ballTreeNodes['radius'][self.__relations[index]['parent']] < radius:
                    index = self.__relations[index]['parent']

            return index
        else:
            return None

    def __ComputeNodeLevel(self, index):
        """
        Compute the level of a node in the tree defined by its index.

        If the node has child nodes, the computation is done for them as well. The levels of the node and its children
        are updated in the internal array of levels.

        :param index: The index of the node

        :type index: int
        """
        if not ('parent' in self.__relations[index]):  # Checking if the node is the root
            self.__levels[index] = 0
        else:
            self.__levels[index] = self.__levels[self.__relations[index]['parent']] + 1

        if 'leftChild' in self.__relations[index]:  # Checking if the node has a left child
            self.__ComputeNodeLevel(self.__relations[index]['leftChild'])
        if 'rightChild' in self.__relations[index]:  # Checking if the node has a right child
            self.__ComputeNodeLevel(self.__relations[index]['rightChild'])

    # ------------- PROPERTIES -----------------
    @property
    def numberOfNodes(self):
        """
        Retrieve the total number of nodes in the ball tree

        :return: The number of nodes in the ball tree
        """
        return self.__numNodes

    @property
    def ballTreeLeaves(self):
        """
        Retrieving the indexes of the nodes which are leaves
        :return: A list of indexes for all the nodes which are leaves
        """
        return nonzero(self.__ballTreeNodes['is_leaf'] == 1)[0]

    @property
    def maxLevel(self):
        """
        Retrieve the maximum level of the tree

        :return: maximum level of the tree

        :rtype: int
        """
        return self.__levels.max()

    # @property
    # def X(self):
    #     return self.ToNumpy()[:, 0]
    #
    # @property
    # def Y(self):
    #     return self.ToNumpy()[:, 1]
    #
    # @property
    # def Z(self):
    #     return self.ToNumpy()[:, 2]

    @property
    def Size(self):
        return self.ToNumpy().shape[0]

    # ------------ GENERAL FUNCTIONS------------------
    def getNodeRadius(self, index):
        """
        Retrieve the radius of a node given by its index

        :param index: The index of the node whose radius is requested

        :type index: int

        :return: The radius of the node

        :rtype: float
        """
        return self.__ballTreeNodes['radius'][index]

    def getPointsOfNode(self, index):
        """
        Retrieve the indexes of the points in a given node

        :param index: The index of the node whose points are required

        :type index: int

        :return: list of indexes of the points in the node

        :rtype: list
        """

        return self.__ballTreePointIndexes[list(range(self.__ballTreeNodes[index]['idx_start'],
                                                      self.__ballTreeNodes[index]['idx_end']))]

    def getSmallestNodesOfSize(self, radius, startingNode='root'):
        """
        Get a list of the smallest nodes whose radii are larger than a given radius

        :param radius: The minimal required size of the nodes
        :param startingNode: Starting node for the search of the smallest node. Can be 'root' (top-down) or 'leaves' (bottom-up). Default: 'root'

        :type radius: float
        :type startingNode: str

        :return: list of the smallest nodes indices (ints)

        :rtype: list (int), None
        """
        if startingNode == 'root':
            return self.__getFirstNodeOfSize(0, radius=radius, startingNode='root')
        elif startingNode == 'leaves':
            leavesIndexes = self.ballTreeLeaves
            return unique(list(map(partial(self.__getFirstNodeOfSize, radius=radius), leavesIndexes)))
        else:
            return None

    def getNodeLevel(self, index):
        """
        Retrieve the level of a node defined by its index

        :param index: The index of the node whose level is required

        :type index: int

        :return: The level of the node

        :rtype: int
        """
        return self.__levels[index]

    def query(self, pnts, k, return_distnaces=False):
        """
        Query the ball tree for the k nearest neighbors of a given set of points

        :param pnts: The query points
        :param k: The number of neighbors to find for the point
        :param return_distnaces: flag to return computed distances

        :type pnts: np.array nx3
        :type k: int

        :return: The indexes for the neighbors of the points

        :rtype: list of np.array

        .. note::
            Return the query points themselves as the first index of each list

        """
        distances, indexes = self.data.query(pnts, k=k)
        if return_distnaces:
            return indexes, distances
        else:
            return indexes

    def queryRadius(self, pnts, radius):
        """
        Query the ball tree to find the neighbors of a given set of point inside a given radius

        :param pnts: The query points
        :param radius: The query radius

        :type pnts: np.array nx3
        :type radius: float

        :return: The indexes for the neighbors of the points

        :rtype: list of np.array

        .. note::
            Return the query points themselves as the first index of each list

        """
        if isinstance(pnts, list):
            pnts = array(pnts)

        if pnts.ndim == 1:
            indexes = self.data.query_radius(pnts.reshape((1, -1)), radius)

            if indexes.dtype == object:
                indexes = indexes[0]

        else:
            indexes = self.data.query_radius(pnts, radius)

        return indexes

    def getLeftChildOfNode(self, index):
        """
        Retrieve the index of the left child node of a given node

        :param index: The index of the node whose left child is required

        :type index: int

        :return: The index of the left child node or None, if the node does not have a left child

        :rtype: int, None

        """
        if 'leftChild' in self.__relations[index]:
            return self.__relations[index]['leftChild']
        else:
            return None

    def getRightChildOfNode(self, index):
        """
        Retrieve the index of the right child node of a given node

        :param index: The index of the node whose right child is required

        :type index: int

        :return: The index of the right child node or None, if the node does not have a right child

        :rtype: int, None
        """
        if 'rightChild' in self.__relations[index]:
            return self.__relations[index]['rightChild']
        else:
            return None

    def getParentOfNode(self, index):
        """
        Retrieve the index of the parent node of a given node

        :param index: The index of the node whose parent is required

        :type index: int

        :return: The index of the parent node or None, if the node does not have a parent (node is the root)

        :rtype: int, None
        """
        if 'parent' in self.__relations[index]:
            return self.__relations[index]['parent']
        else:
            return None

    def getSiblingOfNode(self, index):
        """
        Retrieve the index of the sibling node of a given node

        :param index: The index of the node whose sibling is required

        :type index: int

        :return: The index of the sibling node or None, if the node does not have a sibling

        :rtype: int, None
        """
        if 'sibling' in self.__relations[index]:
            return self.__relations[index]['sibling']
        else:
            return None

    def ToNumpy(self):
        """
        Points as numpy

        """
        return np.asarray(self.data.get_arrays()[0])

    def GetPoint(self, index):
        return self.ToNumpy()[index, :]





