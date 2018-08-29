# -*- coding: UTF-8 -*-
"""
A union-find disjoint set data structure.

"""

# 2to3 sanity
from __future__ import (
    absolute_import, division, print_function, unicode_literals,
)

# Third-party libraries
import numpy as np
import os
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt

####################################################################################################################################

class UnionFind(object):
    """Union-find disjoint sets datastructure.

    Union-find is a data structure that maintains disjoint set
    (called connected components or components in short) membership,
    and makes it easier to merge (union) two components, and to find
    if two elements are connected (i.e., belong to the same
    component).

    This implements the "weighted-quick-union-with-path-compression"
    union-find algorithm.  Only works if elements are immutable
    objects.

    Worst case for union and find: :math:`(N + M \log^* N)`, with
    :math:`N` elements and :math:`M` unions. The function
    :math:`\log^*` is the number of times needed to take :math:`\log`
    of a number until reaching 1. In practice, the amortized cost of
    each operation is nearly linear [1]_.

    Terms
    -----
    Component
        Elements belonging to the same disjoint set

    Connected
        Two elements are connected if they belong to the same component.

    Union
        The operation where two components are merged into one.

    Root
        An internal representative of a disjoint set.

    Find
        The operation to find the root of a disjoint set.

    Parameters
    ----------
    elements : NoneType or container, optional, default: None
        The initial list of elements.

    Attributes
    ----------
    n_elts : int
        Number of elements.

    n_comps : int
        Number of distjoint sets or components.

    Implements
    ----------
    __len__
        Calling ``len(uf)`` (where ``uf`` is an instance of ``UnionFind``)
        returns the number of elements.

    __contains__
        For ``uf`` an instance of ``UnionFind`` and ``x`` an immutable object,
        ``x in uf`` returns ``True`` if ``x`` is an element in ``uf``.

    __getitem__
        For ``uf`` an instance of ``UnionFind`` and ``i`` an integer,
        ``res = uf[i]`` returns the element stored in the ``i``-th index.
        If ``i`` is not a valid index an ``IndexError`` is raised.

    __setitem__
        For ``uf`` and instance of ``UnionFind``, ``i`` an integer and ``x``
        an immutable object, ``uf[i] = x`` changes the element stored at the
        ``i``-th index. If ``i`` is not a valid index an ``IndexError`` is
        raised.

    .. [1] http://algs4.cs.princeton.edu/lectures/

    """

    def __init__(self, elements=None):
        self.n_elts = 0  # current num of elements
        self.n_comps = 0  # the number of disjoint sets or components
        self._next = 0  # next available id
        self._elts = []  # the elements
        self._indx = {}  #  dict mapping elt -> index in _elts
        self._par = []  # parent: for the internal tree structure
        self._siz = []  # size of the component - correct only for roots

        if elements is None:
            elements = []
        for elt in elements:
            self.add(elt)


    def __repr__(self):
        return  (
            '<UnionFind:\n\telts={},\n\tsiz={},\n\tpar={},\nn_elts={},n_comps={}>'
            .format(
                self._elts,
                self._siz,
                self._par,
                self.n_elts,
                self.n_comps,
            ))

    def __len__(self):
        return self.n_elts

    def __contains__(self, x):
        return x in self._indx

    def __getitem__(self, index):
        if index < 0 or index >= self._next:
            raise IndexError('index {} is out of bound'.format(index))
        return self._elts[index]

    def __setitem__(self, index, x):
        if index < 0 or index >= self._next:
            raise IndexError('index {} is out of bound'.format(index))
        self._elts[index] = x

    def add(self, x):
        """Add a single disjoint element.

        Parameters
        ----------
        x : immutable object

        Returns
        -------
        None

        """
        if x in self:
            return
        self._elts.append(x)
        self._indx[x] = self._next
        self._par.append(self._next)
        self._siz.append(1)
        self._next += 1
        self.n_elts += 1
        self.n_comps += 1

    def find(self, x):
        """Find the root of the disjoint set containing the given element.

        Parameters
        ----------
        x : immutable object

        Returns
        -------
        int
            The (index of the) root.

        Raises
        ------
        ValueError
            If the given element is not found.

        """
        if x not in self._indx:
            raise ValueError('{} is not an element'.format(x))

        p = self._indx[x]
        while p != self._par[p]:
            # path compression
            q = self._par[p]
            self._par[p] = self._par[q]
            p = q
        return p

    def connected(self, x, y):
        """Return whether the two given elements belong to the same component.

        Parameters
        ----------
        x : immutable object
        y : immutable object

        Returns
        -------
        bool
            True if x and y are connected, false otherwise.

        """
        return self.find(x) == self.find(y)

    def union(self, x, y):
        """Merge the components of the two given elements into one.

        Parameters
        ----------
        x : immutable object
        y : immutable object

        Returns
        -------
        None

        """
        # Initialize if they are not already in the collection
        for elt in [x, y]:
            if elt not in self:
                self.add(elt)

        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return
        if self._siz[xroot] < self._siz[yroot]:
            self._par[xroot] = yroot
            self._siz[yroot] += self._siz[xroot]
        else:
            self._par[yroot] = xroot
            self._siz[xroot] += self._siz[yroot]
        self.n_comps -= 1

    def component(self, x):
        """Find the connected component containing the given element.

        Parameters
        ----------
        x : immutable object

        Returns
        -------
        set

        Raises
        ------
        ValueError
            If the given element is not found.

        """
        if x not in self:
            raise ValueError('{} is not an element'.format(x))
        elts = np.array(self._elts)
        vfind = np.vectorize(self.find)
        roots = vfind(elts)
        return set(elts[roots == self.find(x)])

    def components(self):
        """Return the list of connected components.

        Returns
        -------
        list
            A list of sets.

        """
        elts = np.array(self._elts)
        vfind = np.vectorize(self.find)
        roots = vfind(elts)
        distinct_roots = set(roots)
        return [set(elts[roots == root]) for root in distinct_roots]
        # comps = []
        # for root in distinct_roots:
        #     mask = (roots == root)
        #     comp = set(elts[mask])
        #     comps.append(comp)
        # return comps

    def component_mapping(self):
        """Return a dict mapping elements to their components.

        The returned dict has the following semantics:

            `elt -> component containing elt`

        If x, y belong to the same component, the comp(x) and comp(y)
        are the same objects (i.e., share the same reference). Changing
        comp(x) will reflect in comp(y).  This is done to reduce
        memory.

        But this behaviour should not be relied on.  There may be
        inconsitency arising from such assumptions or lack thereof.

        If you want to do any operation on these sets, use caution.
        For example, instead of

        ::

            s = uf.component_mapping()[item]
            s.add(stuff)
            # This will have side effect in other sets

        do

        ::

            s = set(uf.component_mapping()[item]) # or
            s = uf.component_mapping()[item].copy()
            s.add(stuff)

        or

        ::

            s = uf.component_mapping()[item]
            s = s | {stuff}  # Now s is different

        Returns
        -------
        dict
            A dict with the semantics: `elt -> component contianing elt`.

        """
        elts = np.array(self._elts)
        vfind = np.vectorize(self.find)
        roots = vfind(elts)
        distinct_roots = set(roots)
        comps = {}
        for root in distinct_roots:
            mask = (roots == root)
            comp = set(elts[mask])
            comps.update({x: comp for x in comp})
            # Change ^this^, if you want a different behaviour:
            # If you don't want to share the same set to different keys:
            # comps.update({x: set(comp) for x in comp})
        return comps

####################################################################################################################################


def find_all_file_name(file_dir, file_type = '.jpg', keyword = ''):  # 特定类型的文件
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if (os.path.splitext(file)[1]).lower() == file_type.lower() and keyword in file:
                L.append((os.path.join(root, file), file))
    print("%d files found in %s\n" %(len(L),file_dir))
    return L
####################################################################################################################################




def save_file(list, filename, filepath = os.getcwd(), filesep = '/'):
    thefile = open(filepath + filesep + filename, 'w')
    for item in list:
        thefile.write("%s\n" %item)

####################################################################################################################################



def load_image(filename, path = '', full_path = True):
    if full_path:
        img = io.imread(filename)
    else:
        img = io.imread(os.path.join(path,filename))
    return img



####################################################################################################################################

def visualize_cluster(clusters, img_path):
    idx = 0
    all_files = find_all_file_name(img_path,'.jpg')
    dict_all = dict()
    # convert to dict:
    for af in all_files:
        dict_all[af[1]] = af[0]
    for cl in clusters:
        plt.figure(idx)
        idx2 = 1
        # print(len(cl))
        for img_name in cl:
            img = io.imread(dict_all[img_name]) if os.path.exists(dict_all[img_name]) else \
	            print("ERROR! NO image called %s in %s" %(dict_all[img_name],img_path))
            plt.subplot(int(np.floor(np.sqrt(len(cl)))), int(np.ceil((len(cl)/int(np.floor(np.sqrt(len(cl))))))), idx2)
            plt.imshow(img)
            idx2 += 1
            plt.axis('off')
        plt.suptitle("Event %d has %d pics" %(idx, len(cl)))
        idx += 1
    plt.show(all)



################################################################################################################################
def save2csv(features_m, file_names, usr_nm, train_ratio, save_path, file_type='original', convert_idx_fn=True):
    # (optional) save it to a file
    # pdb.set_trace()
    columns_name = ['1st Image', '2nd Image', 'Distance', 'Sec', 'Day', 'Sec_in_day', 'Delta_time_freq',
                    'ExposureTime', 'Flash', 'FocalLength', 'ShutterSpeedValue', 'SceneType', 'SensingMethod',
                    'Holiday', 'Delta_closest_holiday', 'Average_closest_holiday', 'Average_city_prop',
                    'Label_e', "Label_s"]
    try:
        df = pd.DataFrame(features_m, columns = columns_name)
        # pdb.set_trace()
        if convert_idx_fn:
            df.loc[:, '1st Image'] = file_names[df['1st Image'].apply(int)]  # convert 1st Image to an int
            df.loc[:, '2nd Image'] = file_names[df['2nd Image'].apply(int)]  # convert 2nd Image to an int

        df.loc[:, 'Day'] = df['Day'].apply(int)  # convert Day to an int
        df.loc[:, 'SceneType'] = df['SceneType'].apply(int)  # convert SceneType to an int
        df.loc[:, 'SensingMethod'] = df['SensingMethod'].apply(int)  # convert SensingMethod to an int
        df.loc[:, 'Label_e'] = df['Label_e'].apply(int)  # convert Label_e to an int
        df.loc[:, 'Label_s'] = df['Label_s'].apply(int)  # convert Label_s to an int
        df.loc[:, 'Holiday'] = df['Holiday'].apply(int)  # convert Holiday to an int
        df.to_csv(os.path.join(save_path, file_type, "%s_%s_data_%1.2f.csv" %(usr_nm, file_type, train_ratio)), header=None, index=False)
        print("Number of %s samples is %d" %(file_type, len(df)))
    except Exception as e:
        print('Error: save %s failed!!!' %file_type)
        print(str(e))
        return os.path.join(save_path, file_type, "%s_%s_data_%1.2f.csv" %(usr_nm, file_type, train_ratio))
    return os.path.join(save_path, file_type, "%s_%s_data_%1.2f.csv" %(usr_nm, file_type, train_ratio))


################################################################################################################################
def combine_csv(name_list, output_nm, common_path):
    """
    name_list = ['hxl', 'hw', 'zzx', 'zt', 'zd', 'wy_tmp', 'lf', 'hhl', 'hxl2016']
    :param name_list:
    :param output_nm:
    :param common_path:
    :return:
    """
    try:
        if os.path.exists(os.path.join(common_path, output_nm)):
            fout = open(os.path.join(common_path, output_nm), 'w')
        else:
            fout = open(os.path.join(common_path, output_nm), 'a')
        for nm in name_list:
            fnm = open(os.path.join(common_path, nm))
            [fout.write(line) for line in fnm]
        fout.close()
    except Exception as e:
        print(e)
        return False
    return True


####################################################################################################################################
def seperate_train_val(filename, train_size=0.98):
    features_m = (pd.read_csv(filename)).values
    np.random.shuffle(features_m)
    # split into training and validation data set
    sep_idx = int(len(features_m) * train_size)
    return features_m[:sep_idx, :], features_m[sep_idx:, :]



