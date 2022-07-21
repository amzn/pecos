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
import pytest  # noqa: F401; pylint: disable=unused-variable
import scipy.sparse as smat
import numpy as np
from pecos.utils.cluster_util import ClusterChain
from pecos.distributed.diagnostic_tools.test_util import DummyComm


class GenerateClusterChain(object):
    """Class to generate cluster chain for testing purpose."""

    @classmethod
    def gen_cluster_chain(cls, n_cluster, nr_label, nr_splits):
        """Generate cluster chain with given parameters"""

        assert nr_label >= n_cluster, (n_cluster, nr_label)
        max_leaf_size = int(np.ceil(nr_label / n_cluster))

        # Assign labels to leaves so that
        # [0, max_leaf_size) labels belong to the first leaf,
        # [max_leaf_size, max_leaf_size*2) belong to the second leaf...
        leaf_cluster = smat.csc_matrix(
            np.repeat(np.eye(n_cluster, dtype=np.float32), max_leaf_size, axis=0)[:nr_label, :],
            dtype=np.float32,
        )

        return ClusterChain.from_partial_chain(leaf_cluster, nr_splits, nr_splits)


def test_dist_cluster_chain():
    """Test class DistClusterChain"""
    import numpy as np
    from pecos.distributed.xmc.base import DistClusterChain

    n_cluster = 8
    nr_label = 15
    nr_splits = 2
    split_depth = 1

    cluster_chain = GenerateClusterChain.gen_cluster_chain(n_cluster, nr_label, nr_splits)

    # test init
    dist_cluster_chain = DistClusterChain(cluster_chain, split_depth)

    # test get nr_splits, avg_leaf_size, leaf_cluster
    assert dist_cluster_chain.get_nr_splits() == nr_splits
    assert dist_cluster_chain.get_avg_leaf_size() == nr_label // n_cluster
    assert np.array_equal(
        dist_cluster_chain.get_leaf_cluster().toarray(), cluster_chain[-1].toarray()
    )

    # test get meta-tree
    n_meta_leaf_cluster = nr_splits**split_depth
    meta_leaf_size = int(np.ceil(nr_label / n_meta_leaf_cluster))
    meta_leaf_cluster = np.repeat(
        np.eye(n_meta_leaf_cluster, dtype=np.float32), meta_leaf_size, axis=0
    )[:nr_label, :]
    assert np.array_equal(
        meta_leaf_cluster, dist_cluster_chain.get_meta_tree_leaf_cluster().toarray()
    )

    # test get sub-tree
    assert dist_cluster_chain.get_num_sub_tree() == nr_splits**split_depth

    dist_sub_assign_list = dist_cluster_chain.get_sub_tree_assignment()
    label_list = np.arange(nr_label)
    sub_assign_list = [
        label_list[i : i + meta_leaf_size] for i in range(0, nr_label, meta_leaf_size)
    ]
    for sub_assign_arr, dist_sub_assign_arr in zip(sub_assign_list, dist_sub_assign_list):
        assert np.array_equal(sub_assign_arr, dist_sub_assign_arr)

    n_leaf_sub_cluster = n_cluster // dist_cluster_chain.get_num_sub_tree()
    max_leaf_size = int(np.ceil(nr_label / n_cluster))
    for idx in range(dist_cluster_chain.get_num_sub_tree()):
        sub_tree_chain = dist_cluster_chain.get_sub_tree_chain(idx)
        n_sub_label = len(dist_cluster_chain.get_sub_tree_assignment(idx))
        # check leaf cluster
        assert np.array_equal(
            sub_tree_chain[-1].toarray(),
            np.repeat(np.eye(n_leaf_sub_cluster, dtype=np.float32), max_leaf_size, axis=0)[
                :n_sub_label, :
            ],
        )

    # test assemble from meta-tree and sub-trees
    meta_tree_chain = dist_cluster_chain.get_meta_tree_chain()
    meta_leaf_cluster = dist_cluster_chain.get_meta_tree_leaf_cluster()
    meta_full_chain = ClusterChain([*meta_tree_chain[:], meta_leaf_cluster])

    all_sub_chain_list = [
        dist_cluster_chain.get_sub_tree_chain(idx)
        for idx in range(dist_cluster_chain.get_num_sub_tree())
    ]

    assembled_dist_chain = DistClusterChain.assemble_from_meta_sub_chain(
        meta_full_chain, all_sub_chain_list
    )

    assert assembled_dist_chain.get_split_depth() == dist_cluster_chain.get_split_depth()
    assert len(assembled_dist_chain.get_cluster_chain()) == len(
        dist_cluster_chain.get_cluster_chain()
    )
    assert np.array_equal(
        assembled_dist_chain.get_leaf_cluster().toarray(),
        dist_cluster_chain.get_leaf_cluster().toarray(),
    )
    for assemble_arr, dist_arr in zip(
        assembled_dist_chain.get_cluster_chain(), dist_cluster_chain.get_cluster_chain()
    ):
        assert np.array_equal(assemble_arr.toarray(), dist_arr.toarray())

    # test new instance re-split at new depth
    re_split_dist_chain = dist_cluster_chain.new_instance_re_split(min_n_sub_tree=4)
    assert re_split_dist_chain.get_num_sub_tree() == 4
    assert re_split_dist_chain.get_split_depth() == np.log2(4)

    with pytest.raises(ValueError):
        dist_cluster_chain.new_instance_re_split(min_n_sub_tree=100)


def test_dist_clustering():
    """Test class DistClustering"""
    from pecos.distributed.xmc.base import DistClustering
    from pecos.xmc.base import HierarchicalKMeans

    nr_label = 10

    dummy_comm = DummyComm()
    cluster_params = DistClustering.ClusterParams(
        indexer_params=HierarchicalKMeans.TrainParams(
            nr_splits=2, max_leaf_size=2, threads=1, seed=0
        ),
        meta_label_embedding_method="pifa",
        sub_label_embedding_method="pifa",
    )
    X = smat.csr_matrix(np.eye(nr_label), dtype=np.float32)
    Y = smat.csc_matrix(np.eye(nr_label), dtype=np.float32)

    dist_cluster_chain = DistClustering(dummy_comm, cluster_params).dist_get_cluster_chain(X, Y)
    assert dist_cluster_chain.get_split_depth() == 1  # Single machine
    assert dist_cluster_chain.get_num_sub_tree() == 2
    assert np.array_equal(
        dist_cluster_chain.get_leaf_cluster().toarray(),
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )
