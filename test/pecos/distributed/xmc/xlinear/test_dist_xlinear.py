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
from pecos.distributed.diagnostic_tools.test_util import DummyComm
from pecos.utils.cluster_util import ClusterChain


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


def test_xlinear_load_balancer():
    """Test class XLinearLoadBalancer"""
    from pecos.distributed.xmc.xlinear.model import XLinearLoadBalancer
    from pecos.distributed.xmc.base import DistClusterChain

    num_machine = 3
    main_workload_factor = 0.95
    threads = 1
    nr_label = 150

    dist_cluster_chain = DistClusterChain(
        cluster_chain=GenerateClusterChain.gen_cluster_chain(
            n_cluster=8, nr_label=nr_label, nr_splits=2
        ),
        split_depth=2,
    )

    Y = smat.csc_matrix(np.eye(nr_label), dtype=np.float32)
    load_balancer = XLinearLoadBalancer(num_machine, main_workload_factor, threads)

    (
        main_job,
        worker_job_list,
        worker_recv_order_list,
    ) = load_balancer.get_load_balanced_sub_train_jobs_list(dist_cluster_chain, Y)

    assert np.array_equal(main_job.sub_tree_idx_arr, np.array([2]))
    assert np.array_equal(worker_job_list[0].sub_tree_idx_arr, np.array([0, 3]))
    assert np.array_equal(worker_job_list[1].sub_tree_idx_arr, np.array([1]))
    assert np.array_equal(worker_recv_order_list, [2, 1])


def test_dist_training():
    """Test class DistTraining and DistributedCPUXLinearModel"""
    from pecos.xmc.base import HierarchicalKMeans
    from pecos.xmc import MLModel, HierarchicalMLModel
    from pecos.distributed.xmc.base import DistClustering
    from pecos.distributed.xmc.xlinear.model import DistributedCPUXLinearModel

    nr_label = 10
    depth = 4

    dummy_comm = DummyComm()

    cluster_params = DistClustering.ClusterParams(
        indexer_params=HierarchicalKMeans.TrainParams(
            nr_splits=2, max_leaf_size=2, threads=1, seed=0
        ),
        meta_label_embedding_method="pifa",
        sub_label_embedding_method="pifa",
    )
    train_params = DistributedCPUXLinearModel.TrainParams(
        hlm_args=HierarchicalMLModel.TrainParams(
            model_chain=[MLModel.TrainParams() for _ in range(depth)]
        )
    )
    pred_params = DistributedCPUXLinearModel.PredParams(
        hlm_args=HierarchicalMLModel.PredParams(
            model_chain=[MLModel.PredParams() for _ in range(depth)]
        )
    )
    dist_params = DistributedCPUXLinearModel.DistParams(
        min_n_sub_tree=2, main_workload_factor=1, threads=1
    )

    X = smat.csr_matrix(np.eye(nr_label), dtype=np.float32)
    Y = smat.csc_matrix(np.eye(nr_label), dtype=np.float32)

    xlm = DistributedCPUXLinearModel.train(
        dummy_comm, X, Y, cluster_params, train_params, pred_params, dist_params
    )

    assert len(xlm.model.model_chain) == depth

    C = xlm.model.model_chain[-1].C
    W = xlm.model.model_chain[-1].W
    assert np.array_equal(
        smat.csc_matrix((C.data, C.indices, C.indptr), shape=C.shape).toarray(),
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
            ]
        ),
    )
    assert np.allclose(
        smat.csc_matrix((W.data, W.indices, W.indptr), shape=W.shape).toarray(),
        np.array(
            [
                [0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.65984, 0.0, 0.0, 0.0, 0.0, 0.0, -0.65984],
                [0.0, 0.0, 0.0, 0.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.40, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.65984, -0.65984, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.663936, 0.663936, 0.0],
                [0.0, 0.0, 0.0, -0.663936, 0.0, 0.0, 0.0, 0.0, 0.0, 0.663936],
                [0.40, 0.40, 0.40, 0.0, 0.40, 0.40, 0.40, 0.0, 0.0, 0.0],
            ]
        ),
    )
