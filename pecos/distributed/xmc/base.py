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
import logging
import numpy as np
import dataclasses as dc
import itertools
from pecos.utils import smat_util
from pecos.xmc import Indexer, LabelEmbeddingFactory
from pecos.xmc.base import HierarchicalKMeans
from pecos.core import clib
from pecos.utils.cluster_util import ClusterChain
from scipy.sparse import csr_matrix, csc_matrix
from pecos.distributed.comm.abs_dist_comm import DistComm
from pecos.utils.profile_util import MemInfo


LOGGER = logging.getLogger(__name__)


class DistClusterChain(object):
    """Distributed Cluster Chain

    A class that encapsulates the distributed logic for cluster chain,
    i.e. meta/sub-tree division, and how to assemble back to an ordinary ClusterChain

    Attributes:
        cluster_chain (`ClusterChain`): The full-tree cluster chain.
        split_depth (int): The depth to split meta/sub-trees,
            i.e. the upper `cluster_chain`[:`split_depth`] layers of cluster_chain is meta-tree,
            and the `cluster_chain`[`split_depth`:] layers are sub-trees.
            The `split_depth`-1 layer is the roots for sub-trees.
    """

    def __init__(self, cluster_chain, split_depth):
        assert isinstance(cluster_chain, ClusterChain), type(cluster_chain)
        assert (
            len(cluster_chain) >= 2
        ), f"Cluster Chain should have at least 2 layers, got {len(cluster_chain)}."
        assert isinstance(split_depth, int), type(split_depth)
        assert (
            1 <= split_depth < len(cluster_chain)
        ), f"Split depth should be at least 1, and at most len(cluster_chain)-1, got {split_depth}."

        self._cluster_chain = cluster_chain
        self._split_depth = split_depth

        self._meta_tree_leaf_cluster = None

    def get_cluster_chain(self):
        """Return the cluster chain"""
        return self._cluster_chain

    def get_split_depth(self):
        """Return the split depth"""
        return self._split_depth

    def get_leaf_cluster(self):
        """Return the last layer of cluster chain, i.e. the leaf cluster"""
        return self._cluster_chain[-1]

    def get_nr_splits(self):
        """Deduce nr_splits from cluster chain"""
        if len(self._cluster_chain) <= 2:
            # TODO: this is min_code actually
            # This function now is only used for estimate training main node workloads
            # So it is good for now
            # We might want to enable min_code option for dist_clustering later
            return self._cluster_chain[0].shape[0]

        return self._cluster_chain[1].shape[0] // self._cluster_chain[0].shape[0]

    def get_avg_leaf_size(self):
        """Deduce average leaf size from cluster chain"""
        return self._cluster_chain[-1].shape[0] // self._cluster_chain[-2].shape[0]

    def get_meta_tree_chain(self):
        """Return the meta-tree as a ClusterChain

        i.e. without the last layer: meta-tree leaf cluster.

        Returns:
            meta_tree_chain (`ClusterChain`): The ClusterChain for meta-tree
        """
        meta_tree_chain = ClusterChain(self._cluster_chain[: self._split_depth])

        return meta_tree_chain

    def get_meta_tree_leaf_cluster(self):
        """Recover meta-tree's leaf clusters from full chain and split depth.

        Returns:
            meta_tree_leaf_cluster (sparse matrix): Sparse matrix representing leaf clusters for meta-tree.
        """
        if self._meta_tree_leaf_cluster is None:
            meta_tree_leaf_cluster = self.get_leaf_cluster()
            for curr_d in reversed(range(self._split_depth, len(self._cluster_chain) - 1)):
                meta_tree_leaf_cluster = clib.sparse_matmul(
                    meta_tree_leaf_cluster, self._cluster_chain[curr_d]
                )
            LOGGER.info(f"meta_tree_leaf_cluster: {meta_tree_leaf_cluster.shape}")
            self._meta_tree_leaf_cluster = meta_tree_leaf_cluster

        return self._meta_tree_leaf_cluster

    def get_num_sub_tree(self):
        """Return the number of sub-trees

        The `split_depth` layer is actually the first layer of sub-trees,
        so the length of its parent (represented by n_col) is the number of sub-tree roots,
        i.e. number of sub-trees.
        """
        return self._cluster_chain[self._split_depth].shape[1]

    def get_sub_tree_chain(self, sub_tree_idx):
        """Get the sub-tree as a `ClusterChain` with given index"""
        assert isinstance(sub_tree_idx, (int, np.int64)), type(sub_tree_idx)
        assert (
            0 <= sub_tree_idx < self.get_num_sub_tree()
        ), f"{sub_tree_idx} is not in [{0}, {self.get_num_sub_tree()})"

        nr_splits = self.get_nr_splits()
        leaf_cluster = self.get_leaf_cluster().tocsr()
        sub_tree_assign_arr = self.get_sub_tree_assignment(sub_tree_idx)

        sub_leaf_cluster = leaf_cluster[sub_tree_assign_arr, :]  # csr
        remain_clusters = np.unique(sub_leaf_cluster.indices)  # nonzeor col

        sub_leaf_cluster.tocsc()
        sub_leaf_cluster = sub_leaf_cluster[:, remain_clusters]
        sub_tree_chain = ClusterChain.from_partial_chain(sub_leaf_cluster, nr_splits, nr_splits)

        return sub_tree_chain

    def get_sub_tree_assignment(self, sub_tree_idx=None):
        """Calculate sub-tree label assignment list from last meta cluster chain
        by extracting the non-zero indices for each column (i.e. sub-tree indicator).

        Returns:
            if `sub_tree_idx` is None:
                list of ndarray [a_1, a_2, a_3, ...], where a_i is an array containing
                the assignments of label ids of sub-tree i.
            else:
                ndarray a_i where a_i is the same as above.
        """
        meta_tree_leaf_cluster = self.get_meta_tree_leaf_cluster()
        sub_tree_assignment_list = smat_util.get_csc_col_nonzero(meta_tree_leaf_cluster)
        if sub_tree_idx is not None:
            assert isinstance(sub_tree_idx, (int, np.int64)), type(sub_tree_idx)
            assert (
                0 <= sub_tree_idx < self.get_num_sub_tree()
            ), f"{sub_tree_idx} is not in [{0}, {self.get_num_sub_tree()})"
            return sub_tree_assignment_list[sub_tree_idx]

        return sub_tree_assignment_list

    @classmethod
    def assemble_from_meta_sub_chain(cls, meta_cluster_chain, sub_cluster_chain_list):
        """Assemble the distributed cluster chain from meta/sub-tree cluster chain

        Parameters:
            meta_cluster_chain (`ClusterChain`): The meta-tree cluster chain
            sub_cluster_chain_list (list): List of sub-tree cluster chains
        """
        assert isinstance(meta_cluster_chain, ClusterChain), type(meta_cluster_chain)
        assert isinstance(sub_cluster_chain_list, list), type(sub_cluster_chain_list)

        LOGGER.info(f"Starts assmebling cluster chain..." f" {MemInfo.mem_info()}")

        # Extract meta-tree from cluster chain without the last layer, i.e. leaf cluster
        cluster_chain = [meta_cluster_chain[i] for i in range(len(meta_cluster_chain) - 1)]
        # split_depth is the depth of meta-tree
        split_depth = len(cluster_chain)

        # Get concatenated sub-tree label assignment array
        sub_tree_assignment_arr = np.concatenate(
            smat_util.get_csc_col_nonzero(meta_cluster_chain[-1])
        )

        # Assemble sub tree layers
        n_sub_depth = len(sub_cluster_chain_list[0])
        for cur_d in range(n_sub_depth):
            # Extract all csc for the layer
            sub_tree_chains = [sub_chain[cur_d] for sub_chain in sub_cluster_chain_list]
            # Diagonal stack csc to get the full layer
            cur_chain = smat_util.block_diag_csc(sub_tree_chains)
            # last layer
            if cur_d == n_sub_depth - 1:
                cur_chain.indices = sub_tree_assignment_arr[cur_chain.indices]
            cluster_chain.append(cur_chain)

        LOGGER.info(
            f"Done assmebling cluster chain. Split depth: {split_depth}. Chain length: {len(cluster_chain)}"
            f" {MemInfo.mem_info()}"
        )

        return cls(ClusterChain(cluster_chain), split_depth)

    def new_instance_re_split(self, min_n_sub_tree):
        """Re-split the cluster by returning a new instance
        with new `split_depth` calculated from given minimum number of sub-trees.

        The new split depth is defined as the smallest depth of the meta-tree
        to have >=`min_n_sub_tree` sub-trees.
        i.e. the parent of the `split_depth` layer should be the shallowest layer
        to have >=`min_n_sub_tree` clusters.

        Parameters:
            min_n_sub_tree (int): The minimum for number of sub-trees in the new split `DistClusterChain`.
        """
        assert isinstance(min_n_sub_tree, int), type(min_n_sub_tree)
        assert min_n_sub_tree >= 1, min_n_sub_tree

        if len(self._cluster_chain) <= 1:
            raise ValueError("Cluster chain should at least have 2 layers to split.")

        if self._cluster_chain[-1].shape[1] < min_n_sub_tree:
            raise ValueError(
                f"Last layer's parent has fewer clusters: {self._cluster_chain[-1].shape[1]} "
                f"than minimum number of sub-tree: {min_n_sub_tree}, cannot split."
            )

        for new_split_depth in range(1, len(self._cluster_chain)):
            if self._cluster_chain[new_split_depth].shape[1] >= min_n_sub_tree:
                break

        return self.__class__(self._cluster_chain, new_split_depth)

    def get_meta_Y(self, Y, threads):
        """Calculate Y for meta tree training"""
        meta_cluster_leaf_cluster = self.get_meta_tree_leaf_cluster()
        meta_Y = clib.sparse_matmul(Y, meta_cluster_leaf_cluster, threads=threads).tocsc()

        return meta_Y

    @classmethod
    def broadcast(cls, dist_comm, dist_cluster_chain, root=0):
        """Broadcast self to all nodes from root.

        Parameters:
            dist_comm (DistComm): Distributed communicators.
            dist_cluster_chain (None or cls): Distributed cluster chain to broadcast.
                On root node, it is cls; On other nodes, it is None.
            root (int): Root node rank for broadcasting.
        """
        if dist_comm.get_rank() == root:
            LOGGER.info(f"Broadcasting distributed cluster chain from Node {root}...")

        split_depth = None
        cluster_chain = None
        # root node to broadcast from, dist_cluster_chain is not None
        if dist_comm.get_rank() == root:
            assert isinstance(dist_cluster_chain, cls), type(dist_cluster_chain)
            split_depth = dist_cluster_chain.get_split_depth()
            cluster_chain = dist_cluster_chain.get_cluster_chain()

        split_depth = dist_comm.bcast(split_depth, root=root)
        cluster_chain = dist_comm.bcast(cluster_chain, root=root)

        if dist_comm.get_rank() == root:
            LOGGER.info(f"Done broadcast distributed cluster chain from Node {root}.")

        return cls(cluster_chain, split_depth)


class DistClustering(object):
    """Distributed clustering

    A class distributedly generate clustering for given data.
    """

    @dc.dataclass
    class ClusterParams(object):
        """Clustering parameters of Distributed Cluster Chain

        Parameters:
            indexer_params (HierarchicalKMeans.TrainParams): Params for indexing
            meta_label_embedding_method (str): Meta-tree cluster label embedding method
            sub_label_embedding_method (str): Sub-tree cluster label embedding method
        """

        indexer_params: HierarchicalKMeans.TrainParams  # type: ignore
        meta_label_embedding_method: str
        sub_label_embedding_method: str

    def __init__(self, dist_comm, cluster_params):
        assert isinstance(dist_comm, DistComm), type(dist_comm)
        assert isinstance(cluster_params, self.ClusterParams), type(cluster_params)
        assert cluster_params.meta_label_embedding_method in (
            "pii",
            "pifa",
        ), cluster_params.meta_label_embedding_method
        assert cluster_params.sub_label_embedding_method in (
            "pii",
            "pifa",
        ), cluster_params.sub_label_embedding_method

        self._dist_comm = dist_comm
        self._indexer_params = cluster_params.indexer_params
        self._meta_embedding_method = cluster_params.meta_label_embedding_method
        self._sub_embedding_method = cluster_params.sub_label_embedding_method

    def _get_meta_tree_max_leaf_size(self, nr_labels):
        """Determine meta tree max leaf size from number of distributed machines and number of labels.

        The formula for calculating meta-tree depth from max_leaf_size is:
            meta_depth = max(1, int(ceil(log2(nr_labels / max_leaf_size))))
        it is also the split depth for distributed cluster chain.
        """

        # Get number of sub-trees for meta-cluster
        # Starting from full tree number of leaf clusters
        num_sub_trees = 2 ** max(
            1, int(np.ceil(np.log2(nr_labels / self._indexer_params.max_leaf_size)))
        )
        while num_sub_trees > self._indexer_params.nr_splits:
            # bottom up
            num_sub_trees //= self._indexer_params.nr_splits
        while num_sub_trees < self._dist_comm.get_size():
            # Make sure each node get at least one tree
            num_sub_trees *= self._indexer_params.nr_splits
        LOGGER.info(
            f"Determined meta-tree leaf clusters number: {num_sub_trees}. "
            f"{self._dist_comm.get_size()} nodes will train {num_sub_trees} sub-trees. "
            f"Number of data labels: {nr_labels}, nr_splits: {self._indexer_params.nr_splits}"
        )

        # Calculate meta-tree leaf size
        meta_tree_max_leaf_size = int(np.ceil(nr_labels / num_sub_trees)) + 1

        if meta_tree_max_leaf_size < self._indexer_params.max_leaf_size:
            LOGGER.warning(
                f"Meta-tree max leaf size: {meta_tree_max_leaf_size} smaller than "
                f"given param max leaf size: {self._indexer_params.max_leaf_size}, "
                f"will round up to param max leaf size, and some machines will be idle in distributed clustering."
            )
            return self._indexer_params.max_leaf_size

        return meta_tree_max_leaf_size

    def _train_meta_cluster(self, X, Y):
        """Generate Meta-tree cluster"""

        LOGGER.info(
            f"Starts creating label embedding {self._meta_embedding_method.upper()} for meta tree on Rank {self._dist_comm.get_rank()} node..."
            f" {MemInfo.mem_info()}"
        )
        label_feat = LabelEmbeddingFactory.create(Y, X, method=self._meta_embedding_method)
        LOGGER.info(
            f"Done creating label embedding {self._meta_embedding_method.upper()} for meta tree on Rank {self._dist_comm.get_rank()} node."
            f" {MemInfo.mem_info()}"
        )

        LOGGER.info("Starts generating meta tree cluster on main node...")
        meta_indexer_params = self._indexer_params.to_dict()
        meta_indexer_params["max_leaf_size"] = self._get_meta_tree_max_leaf_size(Y.shape[1])
        meta_cluster_chain = Indexer.gen(label_feat, **meta_indexer_params)
        LOGGER.info(f"Done generating meta tree cluster." f" {MemInfo.mem_info()}")

        return meta_cluster_chain

    def _train_sub_clusters(self, self_sub_tree_assign_arr_list, X, Y):
        """Generate a list of sub-tree clusters"""

        LOGGER.info(
            f"Rank {self._dist_comm.get_rank()} get {len(self_sub_tree_assign_arr_list)} sub-tree assignments."
        )

        sub_cluster_chain_list = []
        for idx, sub_tree_assign_arr in enumerate(self_sub_tree_assign_arr_list):
            LOGGER.info(
                f"On rank {self._dist_comm.get_rank()}, {idx}th sub-tree assignment "
                f"has {len(sub_tree_assign_arr)} labels: {sub_tree_assign_arr[:10]}..."
            )
            LOGGER.info(
                f"Starts creating label embedding {self._sub_embedding_method.upper()} for {idx}th sub-tree on rank {self._dist_comm.get_rank()}..."
                f" {MemInfo.mem_info()}"
            )
            sub_Y = Y[:, sub_tree_assign_arr]

            label_feat = LabelEmbeddingFactory.create(sub_Y, X, method=self._sub_embedding_method)
            LOGGER.info(
                f"Done creating label embedding {self._sub_embedding_method.upper()} for {idx}th sub-tree on rank {self._dist_comm.get_rank()}."
                f" {MemInfo.mem_info()}"
            )

            LOGGER.info(
                f"Starts generating {idx}th sub-tree cluster on rank {self._dist_comm.get_rank()}..."
            )
            cluster_chain = Indexer.gen(label_feat, train_params=self._indexer_params)
            LOGGER.info(
                f"Done generating {idx}th sub-tree cluster on rank {self._dist_comm.get_rank()}."
                f" {MemInfo.mem_info()}"
            )

            # Extract the ClusterChain object's chain into list
            sub_cluster_chain_list.append([C for C in cluster_chain])

        return sub_cluster_chain_list

    def _divide_sub_cluster_jobs(self, sub_tree_assign_arr_list):
        """Divide sub-tree clustering jobs into `num_machine` groups

        If len(sub_tree_assign_arr_list) < num_machine, the jobs list is padded with empty group in the end.
        """
        num_machine = self._dist_comm.get_size()

        if len(sub_tree_assign_arr_list) < num_machine:
            LOGGER.warning(
                f"Distributed number of machines: {num_machine} "
                f"larger than number of sub-tree clustering jobs: {len(sub_tree_assign_arr_list)}."
                f"Some machines will be idle."
            )

        # Numpy's array_split pads with empty array if len(sub_tree_assign_arr_list) < num_machine
        grp_list = np.array_split(sub_tree_assign_arr_list, num_machine)

        return [grp.tolist() for grp in grp_list]

    def dist_get_cluster_chain(self, X, Y):
        """Distributed create cluster chain

        Parameters:
            X (csr_matrix(float32)): instance feature matrix of shape (nr_inst, nr_feat)
            Y (csc_matrix(float32)): label matrix of shape (nr_inst, nr_labels)

        Returns:
            cluster_chain (ClusterChain)
        """
        assert isinstance(X, csr_matrix), type(X)
        assert isinstance(Y, csc_matrix), type(Y)

        # Create meta tree cluster chain on main node
        grp_sub_tree_assign_arr_list = None
        if self._dist_comm.get_rank() == 0:
            meta_cluster_chain = self._train_meta_cluster(X, Y)
            # Get sub-tree assignment arrays list for leaf cluster layer of meta-tree
            sub_tree_assign_arr_list = smat_util.get_csc_col_nonzero(meta_cluster_chain[-1])
            # Divide into n_machine groups to scatter
            grp_sub_tree_assign_arr_list = self._divide_sub_cluster_jobs(sub_tree_assign_arr_list)

        # Create sub-tree cluster chain on all nodes
        self_sub_tree_assign_arr_list = self._dist_comm.scatter(
            grp_sub_tree_assign_arr_list, root=0
        )
        sub_cluster_chain_list = self._train_sub_clusters(self_sub_tree_assign_arr_list, X, Y)

        # Assemble the full cluster chain on main node
        all_sub_cluster_chain_list = self._dist_comm.gather(sub_cluster_chain_list, root=0)
        dist_cluster_chain = None
        if self._dist_comm.get_rank() == 0:
            all_sub_cluster_chain_list = list(
                itertools.chain(*all_sub_cluster_chain_list)
            )  # flatten
            dist_cluster_chain = DistClusterChain.assemble_from_meta_sub_chain(
                meta_cluster_chain, all_sub_cluster_chain_list
            )

        # Broadcast
        dist_cluster_chain = DistClusterChain.broadcast(self._dist_comm, dist_cluster_chain, root=0)

        return dist_cluster_chain
