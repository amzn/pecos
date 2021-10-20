#  Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
#  with the License. A copy of the License is located at
#
#  http://aws.amazon.com/apache2.0/
#
#  or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
#  OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
#  and limitations under the License.
import json
import logging
import multiprocessing as mp
import os
from itertools import chain, repeat

import numpy as np
import scipy.sparse as smat
from pecos.utils import smat_util
from pecos.core import clib
from sklearn.preprocessing import normalize as sk_normalize

LOGGER = logging.getLogger(__name__)


class ClusterChain(object):
    """Utility class for representing a hierarchical clustering as a list of CSC matrices.

    Attributes:
        chain (list): A list of sparse matrices that form a hierarchical clustering chain.
    """

    def __init__(self, chain):
        """Initialization

        Args:
            chain (list): A list of sparse matrices that form a valid hierarchical clustering chain.
        """

        if isinstance(chain, type(self)):
            chain = chain.chain
        elif isinstance(chain, smat.spmatrix):
            chain = [chain.tocsc()]

        # check the validity of the clustering chain
        assert isinstance(chain, list), "clustering chain shall be a list of CSC matrices"
        for i in range(len(chain) - 1):
            assert (
                chain[i].shape[0] == chain[i + 1].shape[1]
            ), "matrices do not form a valid hierarchical clustering chain"

        self.chain = chain

    def __delitem__(self, key):
        del self.chain[key]

    def __getitem__(self, key):
        return self.chain[key]

    def __setitem__(self, key, val):
        self.chain[key] = val

    def __iter__(self):
        return iter(self.chain)

    def __len__(self):
        return len(self.chain)

    def __eq__(self, other):
        """
        Returns:
            True if `ClusterChain`s are of same length and their matrices have the same shapes and number of nonzeros, False otherwise.
        """

        if len(self) != len(other):
            return False

        for C_A, C_B in zip(self, other):
            if C_A.shape != C_B.shape or (C_A != C_B).nnz != 0:
                return False
        return True

    def save(self, folder):
        """Save to disk.

        Args:
            folder (str): Folder to save to.
        """

        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as fout:
            fout.write(json.dumps({"len": len(self)}))

        for i, C in enumerate(self):
            smat_util.save_matrix(os.path.join(folder, f"C{i}.npz"), C)

    @classmethod
    def load(cls, path_to_cluster):
        """Load from disk.

        Args:
            path_to_cluster (str): Folder where `ClusterChain` was saved to using `ClusterChain.save`.

        Returns:
            ClusterChain: The loaded object.
        """

        if os.path.isfile(path_to_cluster):
            C = smat_util.load_matrix(path_to_cluster)
            return cls.from_partial_chain(C)

        config_path = os.path.join(path_to_cluster, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Cluster config file, {config_path}, does not exist")

        with open(config_path, "r", encoding="utf-8") as fin:
            config = json.loads(fin.read())
            length = config.get("len", None)
            if length is None:
                raise ValueError(
                    f'Cluster config file, {config_path}, does not have "len" parameter'
                )

        chain = []
        for i in range(length):
            chain.append(
                smat_util.load_matrix(os.path.join(path_to_cluster, f"C{i}.npz"))
                .tocsc()
                .astype(np.float32)
            )

        return cls(chain)

    @classmethod
    def from_partial_chain(cls, C, min_codes=None, nr_splits=16):
        """Creates the clustering matrices necessary for a valid hierarchical clustering chain.

        Except for the bottom level of the hierarchy chain provided, all other levels are filled with dummy CSC matrices (all-one).

        Args:
            C (scipy.sparse.base.spmatrix or list): Partial chain for the bottom level of the hierarchy.
                If sparse matrix, this arg should be the matrix representing the bottom level of the hierarchy.
                If list, this arg should be a list of sparse matrices representing the bottom levels of the hierarchy.
            min_codes (int): The number of direct child nodes that the top level of the hierarchy should have.
            nr_splits (int): The out-degree of the clustering matrices that this method will create.

        Returns:
            ClusterChain: The completed hierarchical clustering chain.
        """
        if min_codes is None:
            min_codes = nr_splits

        if isinstance(C, smat.spmatrix):
            cluster_chain = [C.tocsc()]
        else:
            assert isinstance(C, (cls, list, tuple))
            cluster_chain = C
        cur_C = cluster_chain[0]
        if min_codes is None or min_codes <= 1:
            min_codes = cur_C.shape[1]

        # where cur_C.shape == (nr_labels, nr_codes)
        while cur_C.shape[1] > min_codes:
            nr_codes = cur_C.shape[1]
            cur_codes = np.arange(nr_codes)
            new_codes = cur_codes // nr_splits
            shape = (len(cur_codes), new_codes.max() + 1)
            new_C = smat.csc_matrix(
                (np.ones_like(cur_codes), (cur_codes, new_codes)), shape=shape, dtype=np.float32
            )
            cluster_chain = [new_C] + cluster_chain
            cur_C = new_C

        if cur_C.shape[1] > 1:
            new_C = smat.csc_matrix(np.ones((cur_C.shape[1], 1), dtype=np.float32))
            cluster_chain = [new_C] + cluster_chain
        return cls(cluster_chain)

    def matrix_chain_dimension_check(self, M_dict):
        """Check dimension of matrix chain provided by dictionary with keys being number of layers above leaf elements.

        Args:
            M_dict (dict): dictionary of partial matrix chains to check.

        Returns:
            (nr_insts, nr_labels)
        """
        # get/check the dimensions
        assert isinstance(M_dict, dict)
        nr_labels = self.chain[-1].shape[0]
        assert set(M_dict.keys()) <= set(range(len(self) + 1)), "M_dict got invalid key"
        nr_insts = [v.shape[0] for k, v in M_dict.items() if v is not None]
        assert nr_insts.count(nr_insts[0]) == len(nr_insts), "M_dict first dim do not match"
        nr_insts = nr_insts[0]

        # cannot simply use if M_dict.get(0, None) here
        # since truth value of a [MATRIX/ARRAY] is ambiguous
        if M_dict.get(0, None) is not None:
            assert M_dict[0].shape[1] == self.chain[-1].shape[0]
        for i in range(1, len(self) + 1):
            if M_dict.get(i, None) is not None:
                assert (
                    M_dict[i].shape[1] == self.chain[-i].shape[1]
                ), f"{i}: {M_dict[i].shape}!={self.chain[-i].shape}"

        return nr_insts, nr_labels

    def generate_matching_chain(self, M_dict):
        """Generate a chain of instance to cluster matching matrix for user supplied negative (usn) from partial matching chain.

        Args:
            M_dict (dict): dictionary of partial matching chains, with keys being number of layers above leaf elements.
                M_dict[i].shape[0] == nr_inst, for all i.
                M_dict[0].shape[1] == self.chain[-1].shape[0],
                M_dict[i].shape[1] == self.chain[-i].shape[1], for i >= 1
                M_dict.keys() \\subset range(len(self.chain)+1)

        Returns:
            matching_chain: list of csc matrices for user supplied negatives
        """
        matching_chain = [None] * (len(self) + 1)
        # if nothing is given, return a chain of None
        if M_dict is None or all(M_dict[x] is None for x in M_dict):
            return matching_chain

        nr_insts, nr_labels = self.matrix_chain_dimension_check(M_dict)

        # construct matching chain from incomplete chain
        if M_dict.get(0, None) is not None:
            matching_chain[0] = smat_util.binarized(M_dict[0])
        else:
            matching_chain[0] = smat.csc_matrix((nr_insts, nr_labels), dtype=np.float32)
        for i in range(1, len(self) + 1):
            matching_chain[i] = clib.sparse_matmul(matching_chain[i - 1], self.chain[-i])
            if M_dict.get(i, None) is not None:
                matching_chain[i] += smat_util.binarized(M_dict[i])
            matching_chain[i] = matching_chain[i].tocsc().sorted_indices()
        matching_chain.reverse()

        return matching_chain[:-1]

    def generate_relevance_chain(self, R_dict, norm_type=None, induce=True):
        """Generate a chain of instance to cluster relevance matrix for cost sensitive learning from partial relevance chain.

        Args:
            R_dict (dict): dictionary of partial relevance chains, with keys being number of layers above leaf elements.
                R_dict[i].shape[0] == nr_inst, for all i.
                R_dict[0].shape[1] == self.chain[-1].shape[0],
                R_dict[i].shape[1] == self.chain[-i].shape[1], for i >= 1
                R_dict.keys() \\subset range(len(self.chain)+1)
            norm_type (str, optional): row wise normalziation of resulting relevance matrices. Defatult None to ignore.
                Options: ‘l1’, ‘l2’, ‘max’, 'no-norm', None
            induce (bool, optional): whether to induce missing relevance matrix by label aggregation. Default True

        Returns:
            relevance_chain: list of csc matrices for relevance
        """

        relevance_chain = [None] * (len(self) + 1)
        # if nothing is given, return a chain of None
        if R_dict is None or all(R_dict[x] is None for x in R_dict):
            return relevance_chain

        self.matrix_chain_dimension_check(R_dict)

        # construct relevance chain from incomplete chain
        relevance_chain[0] = R_dict.get(0, None)
        for i in range(1, len(self) + 1):
            if R_dict.get(i, None) is not None:
                relevance_chain[i] = R_dict[i]
            elif relevance_chain[i - 1] is not None and induce:
                relevance_chain[i] = clib.sparse_matmul(relevance_chain[i - 1], self.chain[-i])
            else:
                relevance_chain[i] = None
        relevance_chain.reverse()

        if norm_type not in [None, "no-norm"]:
            relevance_chain = [sk_normalize(rr.tocsr(), norm=norm_type) for rr in relevance_chain]

        return relevance_chain[1:]


def hierarchical_kmeans(
    feat_mat,
    max_leaf_size=100,
    imbalanced_ratio=0,
    imbalanced_depth=100,
    spherical=True,
    seed=0,
    kmeans_max_iter=20,
    threads=-1,
):
    """Python implementation of hierarchical 2-means.

    Args:
        feat_mat (numpy.ndarray or scipy.sparse.csr.csr_matrix): Matrix of label features.
        max_leaf_size (int, optional): The maximum size of each leaf node of the tree. Default is `100`.
        imbalanced_ratio (float, optional): Indicates how relaxed the balancedness constraint of 2-means can be.
            Specifically, if an iteration of 2-means is clustering `L` labels, the size of the output 2 clusters will be within approx `imbalanced_ratio * 2 * L` of each other.
            Value between `0.0` and `0.5` (inclusive). Default is `0.0`.
        imbalanced_depth (int, optional): Maximum depth of imbalanced clustering. After depth `imbalanced_depth` is reached, balanced clustering will be used. Default is `100`.
        spherical (bool, optional): True will l2-normalize the centroids of k-means after each iteration. Default is `True`.
        seed (int, optional): Random seed. Default is `0`.
        kmeans_max_iter (int, optional): Maximum number of iterations for each k-means problem. Default is `20`.
        threads (int, optional): Number of threads to use. `-1` denotes all CPUs. Default is `-1`.

    Returns:
        list: List of CSC matrices representing the generated hierarchical cluster chain.
    """

    global run_kmeans

    def run_kmeans(cluster, c1, c2, min_size, kmeans_max_iter, spherical=True):
        indexer = kmeans(feat_mat_global[cluster], c1, c2, min_size, kmeans_max_iter, spherical)
        return cluster[indexer], cluster[~indexer]

    global kmeans

    def kmeans(feat_mat, c1=-1, c2=-1, min_size=50, kmeans_max_iter=20, spherical=True):
        if c1 == -1:
            c1, c2 = np.random.randint(feat_mat.shape[0]), np.random.randint(1, feat_mat.shape[0])
        c1, c2 = feat_mat[c1], feat_mat[(c1 + c2) % feat_mat.shape[0]]
        old_indexer = np.ones(feat_mat.shape[0]) * -1

        for _ in range(kmeans_max_iter):
            scores = np.squeeze(np.asarray(feat_mat.multiply(c1 - c2).sum(1)))
            indexer = scores >= 0
            if indexer.sum() < min_size:
                indexer = np.zeros(feat_mat.shape[0], dtype=np.bool)
                indexer[np.argpartition(-scores, min_size)[:min_size]] = True
            elif (~indexer).sum() < min_size:
                indexer = np.zeros(feat_mat.shape[0], dtype=np.bool)
                indexer[np.argpartition(scores, min_size)[min_size:]] = True
            if np.array_equal(indexer, old_indexer):
                break
            old_indexer = indexer
            c1 = feat_mat[indexer].sum(0)
            c2 = feat_mat[~indexer].sum(0)
            if spherical:
                c1 = sk_normalize(c1)
                c2 = sk_normalize(c2)
        return indexer

    global feat_mat_global
    feat_mat_global = feat_mat
    random = np.random.RandomState(seed)
    cluster_chain = []
    clusters_big, clusters_small = [], []
    if feat_mat.shape[0] > max_leaf_size:
        clusters_big.append(np.arange(feat_mat.shape[0]))
    else:
        clusters_small.append(np.arange(feat_mat.shape[0]))

    while len(clusters_big) > 0:
        # Do balanced clustering beyond imbalanced_depth to ensure reasonably timely termination
        if len(cluster_chain) >= imbalanced_depth:
            imbalanced_ratio = 0

        num_parent_clusters = len(clusters_big) + len(clusters_small)
        new_clusters_big = []
        cols_big, cols_small = [], [x + len(clusters_big) for x in range(len(clusters_small))]
        seeds = [(random.randint(s), random.randint(1, s)) for s in map(len, clusters_big)]
        min_sizes = [int(s * (0.5 - imbalanced_ratio)) for s in map(len, clusters_big)]

        with mp.Pool(threads if threads > 0 else mp.cpu_count()) as p:
            for col, child_clusters in enumerate(
                p.starmap(
                    run_kmeans,
                    zip(
                        clusters_big,
                        *map(list, zip(*seeds)),
                        min_sizes,
                        repeat(kmeans_max_iter),
                        repeat(spherical),
                    ),
                )
            ):
                for cluster in child_clusters:
                    if len(cluster) > max_leaf_size:
                        new_clusters_big.append(cluster)
                        cols_big.append(col)
                    elif len(cluster) > 0:
                        clusters_small.append(cluster)
                        cols_small.append(col)

        cols = cols_big + cols_small

        cluster_chain.append(
            smat.csc_matrix(
                (np.ones(len(cols)), (range(len(cols)), cols)),
                shape=(len(new_clusters_big + clusters_small), num_parent_clusters),
                dtype=np.float32,
            )
        )

        clusters_big = new_clusters_big

    C = []
    for col, cluster in enumerate(chain(clusters_big, clusters_small)):
        for row in cluster:
            C.append((row, col))
    cluster_chain.append(
        smat.csc_matrix(
            (np.ones(feat_mat.shape[0]), list(map(list, zip(*C)))),
            shape=(feat_mat.shape[0], len(clusters_big) + len(clusters_small)),
            dtype=np.float32,
        )
    )
    return cluster_chain
