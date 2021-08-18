"""
This module contains contains indexing utils for training category
specific models, i.e user supplied top-level clustering of labels.
"""
import sys
import pygtrie
import logging
import numpy as np
import scipy as sp
import scipy.sparse as smat
import multiprocessing as mp
from itertools import chain, repeat
import sklearn.preprocessing as skprep
from sklearn.preprocessing import normalize

from pecos.xmc import Indexer
from pecos.utils import cluster_util
from pecos.xmc.base import HierarchicalKMeans


logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


class TrieWrapper(pygtrie.CharTrie):
    """
    Wrapper around pygtrie.CharTrie for purpose of creating cluster matrix chains
    """

    def get_children(self):
        if isinstance(self._root, pygtrie._Node):
            if isinstance(self._root.children, pygtrie._Children):
                # child_char is char at root for corresponding trie denoted by child_root
                for child_char, child_root in self._root.children.iteritems():
                    child_trie = TrieWrapper()
                    child_trie._root = child_root
                    assert isinstance(child_trie._root, pygtrie._Node)
                    child_trie._sorted = self._sorted
                    yield child_char, child_trie
            elif isinstance(self._root.children, pygtrie._OneChild):
                child_trie = TrieWrapper()
                child_trie._root = self._root.children.node
                child_char = self._root.children.step
                assert isinstance(child_trie._root, pygtrie._Node)
                child_trie._sorted = self._sorted
                yield child_char, child_trie
            else:
                raise Exception(
                    "Not handled child {} of type {}".format(
                        self._root.children, type(self._root.children)
                    )
                )
        else:
            raise Exception(
                "Not handled child {} of type {}".format(
                    self._root.children, type(self._root.children)
                )
            )

    @property
    def n_children(self):
        if isinstance(self._root, pygtrie._Node):
            return len(self._root.children)
        else:
            raise Exception("Not handled root {} of type {}".format(self._root, type(self._root)))

    @property
    def is_leaf(self):
        return self.n_children == 0

    @property
    def n_keys(self):
        return len(self.keys())

    @property
    def n_leaves(self):

        if self.is_leaf:
            return 1
        else:
            n_leaves = 0
            for child_char, child_trie in self.get_children():
                n_leaves += child_trie.n_leaves
            return n_leaves

    def build_cluster_chain(self, depth):

        cluster_chain = self._build_sparse_cluster_chain_helper(depth=depth)

        assert len(cluster_chain) == depth + 1
        # Merge all child cluster chains level wise
        new_chain = []
        for curr_level in range(depth + 1):
            mats_to_merge = cluster_chain[curr_level]

            row_offsets = np.concatenate(([0], np.cumsum([mat.shape[0] for mat in mats_to_merge])))
            col_offsets = np.concatenate(([0], np.cumsum([mat.shape[1] for mat in mats_to_merge])))
            total_n_rows = row_offsets[-1]
            total_n_cols = col_offsets[-1]

            all_rows = [mat.row + row_offset for mat, row_offset in zip(mats_to_merge, row_offsets)]
            all_cols = [mat.col + col_offset for mat, col_offset in zip(mats_to_merge, col_offsets)]

            new_row_idxs = np.concatenate(all_rows)
            new_col_idxs = np.concatenate(all_cols)

            assert len(new_row_idxs) == len(new_col_idxs)
            new_data = np.ones((len(new_row_idxs)))

            new_mat = smat.csr_matrix(
                (new_data, (new_row_idxs, new_col_idxs)),
                shape=(total_n_rows, total_n_cols),
                dtype=sp.float32,
            )
            new_chain.append(new_mat)

        cluster_chain = cluster_util.ClusterChain(new_chain)
        return cluster_chain

    def _build_sparse_cluster_chain_helper(self, depth):
        """
        This version builds a chain of SPARSE cluster matrices in scipy.sparse.coo_matrix format.
        Given a trie, builds cluster chain of length (depth + 1).
        Basically, encode clustering induced by a trie as a chain of sparse matrices,
            where sparse_matrix[i] is of shape [n_nodes_at_level_i_plus_1, n_nodes_at_level_i]
            and leaf_nodes are assumed to be present at level = depth

            While creating cluster chain, the clustering induced by trie is collapsed after given depth is reached
            i.e. after reaching level = depth, the subtree under nodes at level = depth are collapsed into a leaf node
            for purpose of creating a cluster chain

            If given depth is such that a path in trie is shorted than given depth, then the cluster chain is created
            as if the path is extended using a branching factor of 1 upto the required depth

        We assume level 0 contains just 1 node which is the root node so, if depth == 0, then cluster chain
        has just 1 matrix of shape (n_labels, 1)


        Parameters
        ----------
        depth: int
            Depth of trie to consider when building cluster chain i.e. depth at which we assume leaf nodes are present
            This depth can be smaller or greater than depth of trie as long as it is not negative

        Returns
        -------
        List of list of scipy sparse matrices in coo format
            Each list of scipy sparse matrices correspond to sparse matrices for nodes at a particular level of cluster chain
        """
        assert depth >= 0

        if depth == 0:
            # Creating chain for node because of depth constraint
            new_chain = [[smat.coo_matrix(np.ones((self.n_keys, 1)))]]
        elif self.is_leaf:
            # Creating chain for a leaf node
            new_chain = [[smat.coo_matrix(np.ones((self.n_keys, 1)))]] * (depth + 1)
        else:
            all_cluster_chains = []

            if "" in self:  # A substring ends at this node
                par_child_smat = smat.coo_matrix(
                    np.ones((self.n_children + 1, 1))
                )  # 1 for dummy child
                # A substring ends at this node so create a dummy node for that
                dummy_cluster_chain = [[smat.coo_matrix([[1]])]] * (depth)
                all_cluster_chains.append(dummy_cluster_chain)
            else:
                par_child_smat = smat.coo_matrix(np.ones((self.n_children, 1)))

            for child_char, child_trie in self.get_children():
                child_cluster_chain = child_trie._build_sparse_cluster_chain_helper(depth=depth - 1)
                all_cluster_chains += [child_cluster_chain]
                assert len(child_cluster_chain) == depth

            # Merge all child cluster chains level wise
            new_chain = []
            for curr_level in range(depth):
                mats_to_merge = []
                for chain in all_cluster_chains:
                    mats_to_merge += chain[curr_level]
                new_chain += [mats_to_merge]

            new_chain = [[par_child_smat]] + new_chain
            assert len(new_chain) == (depth + 1)

        return new_chain


class TrieIndexer(Indexer):
    @classmethod
    def gen(cls, feat_mat, label_strs=[], depth=1, **kwargs):

        is_sorted = all(label_strs[i] <= label_strs[i + 1] for i in range(len(label_strs) - 1))
        if not is_sorted:
            raise Exception(
                "label_strs should be sorted in order to build a cluster matrices correctly. "
                "If not sorted then rows in last matrix in cluster chain will not correspond to "
                "columns in training- data label matrix "
            )

        LOGGER.info("Starting Trie Indexing")
        trie = TrieWrapper()
        trie.update({lstr: 1 for lstr in label_strs})
        LOGGER.info("Added all labels to trie")
        LOGGER.info("Number of keys = {}".format(trie.n_keys))

        cluster_chain = trie.build_cluster_chain(depth=depth)
        LOGGER.info("Finished building cluster chain")
        return cluster_chain


class HybridIndexer(Indexer):
    @classmethod
    def gen(
        cls,
        feat_mat,
        label_strs=[],
        depth=1,
        spherical=True,
        max_iter=20,
        max_leaf_size=100,
        seed=0,
        **kwargs,
    ):

        # try:
        is_sorted = all(label_strs[i] <= label_strs[i + 1] for i in range(len(label_strs) - 1))
        if not is_sorted:
            raise Exception(
                "label_strs should be sorted in order to build a cluster matrices correctly. "
                "If not sorted then rows in last matrix in cluster chain will not correspond to "
                "columns in training- data label matrix "
            )

        LOGGER.info("Starting Hybrid-Trie Indexing")
        trie = TrieWrapper()
        trie.update({lstr: 1 for lstr in label_strs})
        LOGGER.info("Added all labels to trie. Now building trie till depth = {}".format(depth))

        trie_chain = trie.build_cluster_chain(depth=depth)
        flat_clust = smat.csc_matrix(
            trie_chain.chain[-1]
        )  # Use last mat in chain to define flat clustering

        LOGGER.info("Flat clust shape :{}".format(flat_clust.shape))
        remaining_chain = PreClusteredHierarchicalKMeans.gen(
            feat_mat=feat_mat,
            init_mat=flat_clust,
            hierarchical_codes=False,
            spherical=spherical,
            max_leaf_size=max_leaf_size,
            max_iter=max_iter,
            seed=seed,
        )

        LOGGER.info("Built remaining cluster chain using HC 2-means :".format(flat_clust.shape))
        final_chain = cluster_util.ClusterChain(trie_chain.chain[:-1] + remaining_chain.chain[1:])
        return final_chain


class HKMeans_w_MLC(Indexer):
    KMEANS = 0  # KMEANS
    SKMEANS = 5  # Spherical KMEANS

    @classmethod
    def gen(
        cls,
        feat_mat,
        nr_splits=2,
        max_leaf_size=100,
        imbalanced_ratio=0.0,
        imbalanced_depth=100,
        spherical=True,
        seed=0,
        max_iter=20,
        threads=-1,
        dtype=sp.float32,
        mlc_mats=[],
        use_freq=True,
        **kwargs,
    ):
        if nr_splits != 2:
            raise NotImplementedError

        cluster_chain = hierarchical_kmeans_w_mlc(
            feat_mat=feat_mat,
            mlc_mats=mlc_mats,
            use_freq=use_freq,
            max_leaf_size=max_leaf_size,
            imbalanced_ratio=imbalanced_ratio,
            imbalanced_depth=imbalanced_depth,
            spherical=spherical,
            seed=seed,
            max_iter=max_iter,
            threads=threads,
        )
        cluster_chain = cluster_util.ClusterChain(cluster_chain)
        return cluster_chain

    @staticmethod
    def convert_codes_to_csc_matrix(codes, depth):
        nr_codes = 1 << depth
        nr_elements = len(codes)

        indptr = sp.cumsum(sp.bincount(codes + 1, minlength=(nr_codes + 1)), dtype=sp.uint64)
        indices = sp.argsort(codes * sp.float64(nr_elements) + sp.arange(nr_elements))
        C = smat.csc_matrix(
            (sp.ones_like(indices, dtype=sp.float32), indices, indptr),
            shape=(nr_elements, nr_codes),
        )
        return C


def hierarchical_kmeans_w_mlc(
    feat_mat,
    mlc_mats: list,
    use_freq,
    max_leaf_size=100,
    imbalanced_ratio=0.0,
    imbalanced_depth=100,
    spherical=True,
    seed=0,
    max_iter=20,
    threads=-1,
):
    """

    Parameters
    ----------
    feat_mat
    mlc_mats: list
        list of must link constraint matrix
    use_freq
    max_leaf_size
    imbalanced_ratio
    imbalanced_depth
    spherical
    seed
    max_iter
    threads

    Returns
    -------

    """

    global run_kmeans

    def run_kmeans(cluster, c1, c2, min_size, max_iter, spherical=True):
        if point_freq_global is None:
            indexer = kmeans(feat_mat_global[cluster], None, c1, c2, min_size, max_iter, spherical)
        else:
            indexer = kmeans(
                feat_mat_global[cluster],
                point_freq_global[cluster],
                c1,
                c2,
                min_size,
                max_iter,
                spherical,
            )
        return cluster[indexer], cluster[~indexer]

    global kmeans

    def kmeans(feat_mat, freqs, c1=-1, c2=-1, min_size=50, max_iter=20, spherical=True):
        if c1 == -1:
            c1, c2 = sp.random.randint(feat_mat.shape[0]), sp.random.randint(1, feat_mat.shape[0])
        c1, c2 = feat_mat[c1], feat_mat[(c1 + c2) % feat_mat.shape[0]]
        old_indexer = sp.ones(feat_mat.shape[0]) * -1

        for _ in range(max_iter):
            scores = sp.squeeze(sp.asarray(feat_mat.multiply(c1 - c2).sum(1)))

            if freqs is None:
                indexer = get_split_wo_freq(scores=scores, min_size=min_size)
            else:
                indexer = get_split_w_freq(scores=scores, min_size=min_size, freqs=freqs)

            if sp.array_equal(indexer, old_indexer):
                break
            old_indexer = indexer
            c1 = feat_mat[indexer].sum(0)
            c2 = feat_mat[~indexer].sum(0)
            if spherical:
                c1 = normalize(c1)
                c2 = normalize(c2)
        return indexer

    global feat_mat_global, point_freq_global
    feat_mat_global = feat_mat
    point_freq_global = None

    random = sp.random.RandomState(seed)
    cluster_chain = []
    clusters_big, clusters_small = [], []
    if feat_mat.shape[0] > max_leaf_size:
        clusters_big.append(sp.arange(feat_mat.shape[0]))
    else:
        clusters_small.append(sp.arange(feat_mat.shape[0]))

    while (
        len(clusters_big) > 0
    ):  # Iterate until there is at least one cluster with > max_leaf_size nodes

        curr_level = len(cluster_chain)
        # Do balanced clustering beyond imbalanced_depth to ensure reasonably timely termination
        if curr_level >= imbalanced_depth:
            imbalanced_ratio = 0

        # Enact Must-link constraints by creating connected components based on must-link constraints
        if curr_level >= len(mlc_mats):
            """If there are no must-link constraints for this level onward, then append an identity matrix which
            says that the trivial thing that every point must link to itself!"""
            n = feat_mat.shape[0]
            mlc_mats.append(smat.csr_matrix(smat.diags(np.ones((n)), shape=(n, n))))

        clusters_big_cc = []
        feat_mat_cc = []
        cum_idx_cc = 0
        old_cc_to_new_cc = np.zeros((mlc_mats[curr_level].shape[1])) - 1
        new_cc_to_old_cc = np.zeros((mlc_mats[curr_level].shape[1])) - 1
        num_points_per_cc = []
        for cluster in clusters_big:

            # Get constraints mat and features mat rows for this cluster
            local_feat_mat = feat_mat[cluster]
            local_mlc_mat = mlc_mats[curr_level][cluster]

            # Find # non zero cols in local_mlc_mat. That'll be # conn components(= num_CC) over points in cluster
            num_points = len(cluster)
            non_zero_cols = np.diff(local_mlc_mat.tocsc().indptr).nonzero()[0]
            num_CC = non_zero_cols.shape[0]

            # Retain only non-zero cols in local_mlc_mat. Now it should be of shape num_points x num_CC
            local_mlc_mat = local_mlc_mat[:, non_zero_cols]
            local_num_points_per_cc = np.array(
                np.sum(local_mlc_mat.ceil(), axis=0, dtype=int)
            ).reshape(-1)

            # Get feature vec for each conn component using points in that conn comp.
            # (# conn comp x # points) x (# points x # features) --> ( # conn comp x # features )
            local_feat_mat_w_mlc = local_mlc_mat.transpose() * local_feat_mat
            feat_mat_cc.append(local_feat_mat_w_mlc)
            num_points_per_cc.append(local_num_points_per_cc)

            assert local_mlc_mat.shape == (num_points, num_CC)
            assert local_feat_mat.shape == (num_points, feat_mat.shape[1])
            assert local_feat_mat_w_mlc.shape == (num_CC, feat_mat.shape[1])

            """ Assert that each cols sums to one, and sum of total matrix is equal to num_CC.
             This is important for correctness when getting conn comp vector using point vectors. """
            assert (np.round(np.sum(local_mlc_mat, axis=0)) == np.ones((1, num_CC))).all()
            assert int(np.round(np.sum(local_mlc_mat))) == num_CC

            """ Give indices to each conn comp, offsetting it using cum_idx_cc which keeps track 
                of # conn comp so far, and add this list to cluster_big_cc """
            cc_idxs = np.arange(num_CC) + cum_idx_cc
            clusters_big_cc.append(cc_idxs)

            old_cc_to_new_cc[non_zero_cols] = cc_idxs
            new_cc_to_old_cc[cc_idxs] = non_zero_cols

            cum_idx_cc += num_CC

        feat_mat_global_cc = smat.csr_matrix(smat.vstack(feat_mat_cc))
        if use_freq:
            point_freq_global = np.concatenate(num_points_per_cc).reshape(-1)
            assert point_freq_global.shape == (feat_mat_global_cc.shape[0],)

        clusters_big = clusters_big_cc
        feat_mat_global = feat_mat_global_cc
        LOGGER.info("Shape of new  global feat matrix = {}".format(feat_mat_global.shape))

        num_parent_clusters = len(clusters_big) + len(clusters_small)
        new_clusters_big = []
        new_clusters_small = []
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
                        repeat(max_iter),
                        repeat(spherical),
                    ),
                )
            ):
                for cluster_cc in child_clusters:
                    """cluster is a list of connected component indices.
                    Convert this list to list of indices of points in these connected components"""
                    # Map new conn_comp indices to old conn_comp indices
                    cluster_cc = new_cc_to_old_cc[cluster_cc]

                    # Get mlc matrix with only cols restricted to current list of conn components
                    local_mlc_mat = mlc_mats[curr_level][:, cluster_cc]
                    assert local_mlc_mat.shape == (feat_mat.shape[0], len(cluster_cc))

                    # Get points in these conn components, which have non zero value in their corresponding row
                    cluster = np.diff(local_mlc_mat.indptr).nonzero()[0]
                    if len(cluster) > max_leaf_size and len(cluster_cc) > 1:
                        new_clusters_big.append(cluster)
                        cols_big.append(col)
                    elif len(cluster) > max_leaf_size and len(cluster_cc) == 1:
                        """Add to small clusters, even though this cluster has more than max_leaf_size points
                        because this cluster has just one connected component and thus can not split further due
                        to must-link constraints
                        """
                        new_clusters_small.append(cluster)
                        cols_small.append(col)
                    elif len(cluster) > max_leaf_size and len(cluster_cc) == 0:
                        # This condition is not possible but still having this for a sanity check
                        raise NotImplementedError
                    elif len(cluster) > 0:
                        new_clusters_small.append(cluster)
                        cols_small.append(col)
                    # else: # Do not raise error when a cluster is empty.
                    #     raise NotImplementedError

        cols = cols_big + cols_small

        clusters_small.extend(new_clusters_small)

        curr_clust_mat = smat.csc_matrix(
            (sp.ones(len(cols)), (range(len(cols)), cols)),
            shape=(len(new_clusters_big + clusters_small), num_parent_clusters),
            dtype=sp.float32,
        )
        cluster_chain.append(curr_clust_mat)

        clusters_big = new_clusters_big

        LOGGER.info(
            "Cluster chain shape at level = {} is {}".format(curr_level, curr_clust_mat.shape)
        )

    C = []
    for col, cluster in enumerate(chain(clusters_big, clusters_small)):
        for row in cluster:
            C.append((row, col))

    cluster_mat_cc = smat.csc_matrix(
        (sp.ones(feat_mat.shape[0]), list(map(list, zip(*C)))),
        shape=(feat_mat.shape[0], len(clusters_big) + len(clusters_small)),
        dtype=sp.float32,
    )

    cluster_mat = smat.csc_matrix(mlc_mats[-1] * cluster_mat_cc, dtype=sp.float32)
    cluster_chain.append(cluster_mat)
    LOGGER.info("Cluster chain shape at final level is {}".format(cluster_mat.shape))
    return cluster_chain


def build_prefix_mlc_mat(label_strs, max_pref_len):
    """
    Generates a list must-link constraint (mlc) matrix using a trie. At level d, the constraint is that
    two strings with the exact same prefix upto d chars should be in the same cluster.
    Taking transitive closure of must-link constraints induces connected components at each level.
    Shape of list of mlc matrices is [(# points, # conn_components_at_level_d) for d in range(1, max_pref_len+1)]

    Parameters
    ----------
    label_strs: iterable
        iterable over label strings
    max_pref_len: int
        Max depth upto which I need to create must link constraint cluster matrix using the trie


    Returns
    -------
    List of matrices encoding must link constraints for each level of hierarchical clustering.
    """

    trie_cluster_mat = TrieIndexer.gen(feat_mat=None, label_strs=label_strs, depth=max_pref_len)

    assert len(trie_cluster_mat) == max_pref_len + 1
    for level, mat in enumerate(trie_cluster_mat):
        LOGGER.info("Trie Cluster Matrix: {} {}".format(level, mat.shape))
    LOGGER.info("")

    prefix_mlc_mats = [
        smat.csr_matrix(normalize(trie_cluster_mat[-1], axis=0, norm="l1"))
    ]  # Normalize so that sum of each col is 1
    for pref_len in range(max_pref_len, 0, -1):
        last_mat = trie_cluster_mat[pref_len]
        sec_last_mat = trie_cluster_mat[pref_len - 1]
        new_last_mat = last_mat.dot(sec_last_mat)

        # Remove last 2 matrices and add newly created one
        trie_cluster_mat = trie_cluster_mat[: pref_len - 1] + [new_last_mat]

        # Normalize so that sum of each col is 1
        curr_mlc_mat = smat.csr_matrix(normalize(new_last_mat, axis=0, norm="l1"))
        prefix_mlc_mats.append(curr_mlc_mat)

        assert len(trie_cluster_mat) == pref_len
        LOGGER.info(
            "Multiplying matrix {} with {} to get {}".format(
                sec_last_mat.shape, last_mat.shape, new_last_mat.shape
            )
        )

    prefix_mlc_mats.reverse()  # Reverse the order to go from level 0 to level max_pref_len
    prefix_mlc_mats = prefix_mlc_mats[
        1:
    ]  # Remove first constraint mat because that effectively puts a constraint that everylabel should be one cluster.
    for level, mat in enumerate(prefix_mlc_mats):
        LOGGER.info("MLC Mat {} {}".format(level, mat.shape))
        assert mat.shape[0] == len(label_strs)
        assert int(np.round(np.sum(mat))) == mat.shape[1]
        assert (np.round(np.sum(mat, axis=0)) == np.ones((1, mat.shape[1]))).all()

    assert len(prefix_mlc_mats) == max_pref_len
    return prefix_mlc_mats


def get_split_wo_freq(scores, min_size):

    n = len(scores)
    indexer = scores >= 0  # Default way of assigning points to c1 and c2 by hinging at zero
    if indexer.sum() < min_size:
        indexer = np.zeros(n, dtype=np.bool)
        indexer[sp.argpartition(-scores, min_size)[:min_size]] = True
    elif (~indexer).sum() < min_size:
        indexer = np.zeros(n, dtype=np.bool)
        indexer[sp.argpartition(scores, min_size)[min_size:]] = True

    return indexer


def get_split_w_freq(scores, freqs, min_size):

    total_freqs = freqs.sum()
    n = len(scores)  # Number of points
    indexer = (
        scores >= 0
    )  # First assign points with scores greater than zero to c1 and others to c2

    c1_size = freqs[indexer].sum()
    c2_size = freqs[~indexer].sum()
    assert c1_size + c2_size == total_freqs
    assert c1_size >= min_size or c2_size >= min_size

    if c1_size < min_size:
        indexer = np.zeros(n, dtype=np.bool)
        ordering = np.argsort(
            -1 * scores
        )  # Sort in descending order. Elements towards beginning will be close to c1
        freqsums = np.cumsum(
            freqs[ordering]
        )  # freqs[ordering] gives freqs for points in sorted order of their scores
        part_idx = n - 1
        for i in range(n - 1):
            if freqsums[i] < min_size <= freqsums[i + 1]:
                part_idx = i + 1
                break
        c1_idxs = ordering[: part_idx + 1]
        c2_idxs = ordering[part_idx + 1 :]
        assert freqs[c1_idxs].sum() >= min_size

        if freqs[c2_idxs].sum() < min_size:
            LOGGER.info(
                "WARNING: min_size = {} condition is violaed. c1,c2 sizes = {} ({:.4f}) {} ({:.4f})".format(
                    min_size,
                    freqs[c1_idxs].sum(),
                    freqs[c1_idxs].sum() / total_freqs,
                    freqs[c2_idxs].sum(),
                    freqs[c2_idxs].sum() / total_freqs,
                )
            )

        indexer[c1_idxs] = True
    elif c2_size < min_size:
        indexer = np.zeros(n, dtype=np.bool)
        ordering = np.argsort(
            scores
        )  # Sort in ascending order. Elements towards beginning will be close to c2
        freqsums = np.cumsum(
            freqs[ordering]
        )  # yfreqs[ordering] gives yfreqs for points in sorted order of their scores
        part_idx = n - 1
        for i in range(n - 1):
            if freqsums[i] < min_size <= freqsums[i + 1]:
                part_idx = i + 1
                break
        c2_idxs = ordering[: part_idx + 1]
        c1_idxs = ordering[part_idx + 1 :]
        assert freqs[c2_idxs].sum() >= min_size

        if freqs[c1_idxs].sum() < min_size:
            LOGGER.info(
                "WARNING: min_size = {} condition is violaed. c1,c2 sizes = {} ({:.4f}) {} ({:.4f})".format(
                    min_size,
                    freqs[c1_idxs].sum(),
                    freqs[c1_idxs].sum() / total_freqs,
                    freqs[c2_idxs].sum(),
                    freqs[c2_idxs].sum() / total_freqs,
                )
            )

        indexer[c1_idxs] = True

    return indexer


def reduce_chain_len(cluster_chain, max_depth):
    """

    Parameters
    ----------
    max_depth: int
        Max depth of final cluster chain
    cluster_chain: list
        list of cluster chain
    Returns
    -------
        cluster chain with given max depth
    """

    if isinstance(cluster_chain, cluster_util.ClusterChain):
        cluster_chain = cluster_chain.chain

    assert isinstance(cluster_chain, list)
    n_levels = len(cluster_chain)
    for level in range(n_levels, max_depth, -1):
        last_mat = cluster_chain[level - 1]
        sec_lat_mat = cluster_chain[level - 2]
        new_mat = last_mat.dot(sec_lat_mat)

        cluster_chain = [mat for mat in cluster_chain[: level - 2]] + [new_mat]

    assert len(cluster_chain) == max_depth
    cluster_chain = cluster_util.ClusterChain(cluster_chain)
    return cluster_chain


def print_to_dense(cluster_chain):
    return "\n".join(["{}".format(c.todense()) for c in cluster_chain])


class PreClusteredHierarchicalKMeans(Indexer):
    """Does hierarchical balanced kmeans starting from pre-defined top-level clusters."""

    @classmethod
    def gen(
        cls,
        feat_mat,
        init_mat,
        kdim=2,
        max_leaf_size=100,
        spherical=True,
        seed=0,
        max_iter=20,
        threads=-1,
        hierarchical_codes=True,
        **kwargs,
    ):
        """Main clustering function.

        Parameters:
        ----------
        feat_mat: smat.csr_matrix
            label features to be used for clustering.
        init_mat: smat.csc_matrix
            initial pre-clustering as sparse matrix, rows are labels and columns are codes.
        kdim: int
            number of children for each parent node
        spherical: bool
            true: use spherical kmeans
            false: use regular kmeans
        seed: int
            random seed
        max_iter: int
            max. iterations for clustering
        threads: int
            number of cores to be used
        hierarchical_codes: bool
            true: make a hierarchical tree from roots to codes
            false: root branches into codes directly

        Returns:
        -------
        cluster_chain representing the final clustering.

        """
        if not isinstance(feat_mat, smat.csr_matrix):
            raise ValueError("feat_mat does not follow correct input format")
        if feat_mat.dtype != np.float32:
            raise ValueError("feat_mat does not follow correct data type")
        if not isinstance(init_mat, smat.csc_matrix):
            raise ValueError("init_mat does not follow correct input format")
        if init_mat.dtype != np.float32:
            raise ValueError("init_mat does not follow correct data type")
        label_order = []
        all_cluster_chains = []
        for code in range(init_mat.shape[1]):
            LOGGER.info("Training hierarchical clustering for code: {}".format(code))
            rel_labels = init_mat.indices[init_mat.indptr[code] : init_mat.indptr[code + 1]]
            rel_feat = feat_mat[rel_labels, :]
            all_cluster_chains.append(
                cls.indexer_dict["hierarchicalkmeans"].gen(
                    feat_mat=rel_feat,
                    kdim=kdim,
                    max_leaf_size=max_leaf_size,
                    imbalanced_ratio=0.0000000000000000001,  # Passing non-zero but very very small imbalanced ratio to avoid error thrown by PECOS package when a branch has just a single label.
                    spherical=spherical,
                    seed=seed,
                    threads=threads,
                    max_iter=max_iter,
                )
            )
            label_order += list(rel_labels)

        if hierarchical_codes:
            final_cluster_chain = _index_clusters(feat_mat, init_mat)
        else:
            final_cluster_chain = [
                smat.csc_matrix(
                    np.ones((init_mat.shape[1], 1)),
                    dtype=np.float32,
                )
            ]
        max_depth = max(len(c_chain) for c_chain in all_cluster_chains)
        all_cluster_chains = [
            _extend_to_depth(c_chain, max_depth) for c_chain in all_cluster_chains
        ]
        for d in range(max_depth):
            LOGGER.info("Joining matrices at depth {}".format(d))
            mat_list = [
                all_cluster_chains[cluster][d] for cluster in range(len(all_cluster_chains))
            ]
            final_cluster_chain.append(_block_join(mat_list))
        inverse = [0] * len(label_order)
        for i, p in enumerate(label_order):
            inverse[p] = i
        final_cluster_chain[-1] = final_cluster_chain[-1].tocsr()[inverse, :].tocsc()
        return cluster_util.ClusterChain(final_cluster_chain)


def _extend_to_depth(chain, depth):
    """Extends a cluster chain to a given depth."""
    if depth < len(chain):
        return chain
    num_req = depth - len(chain)
    num_codes = chain[-1].shape[1]
    new_chain = chain[0 : len(chain) - 1]
    for i in range(num_req):
        new_chain.append(smat.identity(num_codes, dtype=np.float32).tocsc())
    new_chain.append(chain[-1])
    return cluster_util.ClusterChain(new_chain)


def _block_join(mat_list):
    """Joins a list of matrices as block diagonals of a larger matrix."""
    row_offset = 0
    col_offset = 0
    data = []
    indices = []
    indptr = []
    prev_len = 0
    for i, mat in enumerate(mat_list):
        data.extend(mat.data)
        indices.extend(mat.indices + row_offset)
        if i < len(mat_list) - 1:
            indptr.extend(mat.indptr[:-1] + prev_len)
        else:
            indptr.extend(mat.indptr + prev_len)
        prev_len += mat.indptr[-1]
        row_offset += mat.shape[0]
        col_offset += mat.shape[1]
    return smat.csc_matrix(
        (data, indices, indptr),
        shape=(row_offset, col_offset),
        dtype=np.float32,
    )


def _index_clusters(feat_mat, init_mat):
    """Creates a hierarchical binary tree till the top-level clusters supplied."""
    cluster_feat = init_mat.T.dot(feat_mat)
    cluster_feat = cluster_feat.tocsr()
    cluster_feat = skprep.normalize(cluster_feat, "l2", axis=1)
    init_cluster = HierarchicalKMeans.gen(
        feat_mat=cluster_feat,
        kdim=2,
        max_leaf_size=2,
        imbalanced_ratio=0.0,
    )
    return init_cluster.chain
