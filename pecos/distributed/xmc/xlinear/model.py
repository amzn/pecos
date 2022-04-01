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
import pecos
from pecos.xmc.xlinear.model import XLinearModel
from pecos.utils import smat_util
from pecos.xmc import MLModel, HierarchicalMLModel
from scipy.sparse import csr_matrix, csc_matrix
from typing import ClassVar
from pecos.distributed.comm.abs_dist_comm import DistComm
from pecos.utils.profile_util import MemInfo
from pecos.distributed.xmc.base import DistClusterChain, DistClustering

LOGGER = logging.getLogger(__name__)


class XLinearLoadBalancer(object):
    """LoadBalancer for XLinear model training to divide sub-tree training jobs into workload-balanced groups

    Attributes:
        n_machine (int): >=1, Number of distributed machines.
        main_workload_factor (float): >0, Factor of main node vs worker node workload,
            in order to decrease main node workload.
        threads (int): Number of threads to use. default -1 to use all.
    """

    @dc.dataclass
    class DistLoadBalancedJob(object):
        """Storage class containing one or more sub-trees' indices for training.
        Facilitates send/receive to machines via distributed communicator.

        Attributes:
            sub_tree_idx_arr (ndarray): The array of sub-tree indices to train in the job
            sub_x_idx_arr_list (list): The list of corresponding instance indices arr
                to slice X for training on each sub-tree
        """

        sub_tree_idx_arr: np.ndarray
        sub_x_idx_arr_list: list

        def get_n_sub_tree(self):
            """Return number of sub-trees to train in this job"""
            return len(self.sub_tree_idx_arr)

        def send(self, dist_comm, dest):
            """Send class attributes to destination rank

            Parameters:
                dist_comm (DistComm): Distributed communicator
                dest (int): MPI rank for destination
            """
            LOGGER.info(
                f"Starts sending sub-training jobs from node {dist_comm.get_rank()} to {dest}..."
            )
            dist_comm.send((self.sub_tree_idx_arr, self.sub_x_idx_arr_list), dest=dest, tag=dest)
            LOGGER.info(
                f"Done sending sub-training jobs from node {dist_comm.get_rank()} to {dest}."
            )

        @classmethod
        def recv(cls, dist_comm, source=0):
            """Receive class atrributes from source on current rank.

            Parameters:
                dist_comm (DistComm): Distributed communicator
                source (int, optional): MPI rank for source, default=0

            Returns:
                cls
            """
            LOGGER.info(
                f"Starts receiving sub-training jobs from source {source} for rank {dist_comm.get_rank()}..."
            )
            sub_tree_idx_arr, sub_x_idx_arr_list = dist_comm.recv(
                source=source, tag=dist_comm.get_rank()
            )
            LOGGER.info(
                f"Done receiving sub-training jobs from source {source} for rank {dist_comm.get_rank()}."
            )

            return cls(sub_tree_idx_arr, sub_x_idx_arr_list)

    def __init__(self, num_machine, main_workload_factor, threads=-1):
        """Initialization"""

        self._num_machine = num_machine
        self._main_workload_factor = main_workload_factor
        self._threads = threads

    def _get_meta_train_workload(self, nr_inst, dist_cluster_chain):
        """Calculate training workload for meta-tree in a heuristic way.

        Returns:
            meta_workload (float): Meta-tree training workload.
        """
        assert isinstance(dist_cluster_chain, DistClusterChain), type(dist_cluster_chain)

        split_depth = dist_cluster_chain.get_split_depth()
        # Sub-tree depth without the leaf clusters layer
        sub_depth = len(dist_cluster_chain.get_cluster_chain()) - split_depth - 1
        nr_splits = dist_cluster_chain.get_nr_splits()
        avg_leaf_size = dist_cluster_chain.get_avg_leaf_size()

        n_meta_negative_samples = nr_splits * split_depth
        n_sub_negative_samples = nr_splits * sub_depth + avg_leaf_size

        LOGGER.info(
            f"meta, sub negative samples: {n_meta_negative_samples} {n_sub_negative_samples}"
        )

        return nr_inst * n_meta_negative_samples / n_sub_negative_samples

    def _get_sub_train_workload_list(self, Y_indptr, sub_tree_assign_arr_list):
        """Calculate training workloads for sub-trees

        Returns:
            sub_workload_list (list of int): List of training workloads for each sub-tree.
        """
        nnz_of_insts = Y_indptr[1:] - Y_indptr[:-1]
        return [
            sum(nnz_of_insts[sub_tree_assign_arr])
            for sub_tree_assign_arr in sub_tree_assign_arr_list
        ]

    def _get_load_balanced_sub_idx_arr_list(self, sub_load_list, baseline_load):
        """Group sub-tree training into given `num_machine` groups in a load-balanced way,
        based on each sub-tree's and baseline workload.

        Returns:
            main_sub_idx_arr (ndarray): Array of sub-tree indices for main node to train.
            worker_sub_idx_arr_list (list): List of arrays of sub-tree indices for worker nodes to train.
            worker_recv_order_list (list): List of the orders for only worker nodes to send/receive jobs.
        """

        # Sort job loads descendingly
        sub_load_idx_list = sorted(
            [(load, i) for i, load in enumerate(sub_load_list)],
            key=lambda tup: tup[0],
            reverse=True,
        )

        # Assign jobs to all nodes to make it balanced
        # Main node workload is weighted by 1/main_workload_factor
        balanced_sub_idx_list = [[] for _ in range(self._num_machine)]
        machine_load_list = [0] * self._num_machine
        machine_load_list[0] = baseline_load / self._main_workload_factor  # For main node
        for sub_load, sub_idx in sub_load_idx_list:
            min_load_id = 0
            for i in range(1, self._num_machine):
                if machine_load_list[i] < machine_load_list[min_load_id]:
                    min_load_id = i
            balanced_sub_idx_list[min_load_id].append(sub_idx)
            if min_load_id == 0:
                # Weighted workload for main node
                machine_load_list[min_load_id] += sub_load / self._main_workload_factor
            else:
                machine_load_list[min_load_id] += sub_load

        # To numpy array
        machine_sub_idx_arr_list = [
            np.array(sub_idx, dtype=int) for sub_idx in balanced_sub_idx_list
        ]

        # Divide main and workers jobs
        main_sub_idx_arr, worker_sub_idx_arr_list = (
            machine_sub_idx_arr_list[0],
            machine_sub_idx_arr_list[1:],
        )
        main_workload, worker_workload_list = machine_load_list[0], machine_load_list[1:]
        # TODO: why we need to receive in workload ascending order?
        # Sort workloads ascendingly for the order to send to worker nodes
        sorted_worker_workload_list = sorted(
            [(workload, idx) for idx, workload in enumerate(worker_workload_list)],
            key=lambda tup: tup[0],
        )
        worker_recv_order_list = [idx + 1 for _, idx in sorted_worker_workload_list]
        LOGGER.info(f"Main node workload: {main_workload}")
        if worker_workload_list:
            LOGGER.info(
                f"Min worker node workload, machine rank: {sorted_worker_workload_list[0]}. "
                f"Max worker node workload, machine rank: {sorted_worker_workload_list[-1]}"
            )
        else:
            LOGGER.info(
                f"Num distributed machine: {self._num_machine}. "
                f"All jobs on main. No worker nodes available."
            )

        return main_sub_idx_arr, worker_sub_idx_arr_list, worker_recv_order_list

    def get_load_balanced_sub_train_jobs_list(self, dist_cluster_chain, Y):
        """Create a Load balanced job list for dividing sub tree training jobs

        Parameters:
            dist_cluster_chain (`DistClusterChain`): Distributed cluster chain.
            Y (csc_matrix(float32)): label matrix of shape (nr_inst, nr_labels)

        Returns:
            main_job (`DistLoadBalancedJob`): Job for main node.
            worker_job_list (list): Jobs for worker nodes.
            worker_recv_order_list (list): List of worker nodes ranks,
                for main node to send/receive the jobs/results.
        """
        assert isinstance(dist_cluster_chain, DistClusterChain), type(dist_cluster_chain)
        assert isinstance(Y, csc_matrix), type(Y)

        meta_workload = self._get_meta_train_workload(Y.shape[0], dist_cluster_chain)
        sub_workload_list = self._get_sub_train_workload_list(
            Y.indptr, dist_cluster_chain.get_sub_tree_assignment()
        )

        (
            main_sub_idx_arr,
            worker_sub_idx_arr_list,
            worker_recv_order_list,
        ) = self._get_load_balanced_sub_idx_arr_list(sub_workload_list, meta_workload)
        LOGGER.info(
            f"Training jobs for all Sub-trees divided onto {self._num_machine} machines: "
            f"Main node will train for {len(main_sub_idx_arr)} sub-trees, "
            f"Worker nodes will train for {[len(worker_arr) for worker_arr in worker_sub_idx_arr_list]} sub-trees, "
            f"worker receive order: {worker_recv_order_list}."
        )

        # Assemble main and worker nodes jobs
        sub_x_idx_arr_list = smat_util.get_csc_col_nonzero(
            dist_cluster_chain.get_meta_Y(Y, self._threads)
        )
        main_job = self.DistLoadBalancedJob(
            sub_tree_idx_arr=main_sub_idx_arr,
            sub_x_idx_arr_list=[
                sub_x_idx_arr_list[cluster_idx] for cluster_idx in main_sub_idx_arr
            ],
        )

        worker_job_list = []
        for worker_sub_idx_arr in worker_sub_idx_arr_list:
            worker_job_list.append(
                self.DistLoadBalancedJob(
                    sub_tree_idx_arr=worker_sub_idx_arr,
                    sub_x_idx_arr_list=[
                        sub_x_idx_arr_list[tree_idx] for tree_idx in worker_sub_idx_arr
                    ],
                )
            )

        return main_job, worker_job_list, worker_recv_order_list


class DistTraining(object):
    """Distributed training with given distributed cluster chain"""

    @dc.dataclass
    class DistSubTreeModel(object):
        """Storage class for only sub-tree model chains to facilitate
        data trasferring across machines with distributed communicator,
        because the original XLinearModel cannot be pickled.

        Attributes:
            sub_tree_idx: Index of the sub-tree for the model
            c_list (list): List of C matrices in XLinearModel.model.model_chain
            w_list (list): List of W matrices in XLinearModel.model.model_chain
            bias_list (list): List of bias in XLinearModel.model.model_chain.
            pred_param_dict_list (list): List of prediction params dict in XLinearModel.model.model_chain.
        """

        # Class variable
        # A very large integer to differentiate between the tags for send/recv model parts
        # Should be larger than total number of sub-trees to send/recv by one worker to make non-dup tags
        # `DistTraining` implements checks for the above requirement.
        # TODO: How do we remove this magic number without exposing external information of
        # one worker's total num of sub-trees to this storage class?
        TAG_OFFSET: ClassVar[int] = 1000000

        sub_tree_idx: int
        c_list: list
        w_list: list
        bias_list: list
        pred_param_dict_list: list

        def send(self, dist_comm, dest):
            """Send class attributes to destination rank

            Parameters:
                dist_comm (DistComm): Distributed communicator
                dest (int): rank for destination
            """
            LOGGER.debug(
                f"Starts sending sub-tree model {self.sub_tree_idx} from node {dist_comm.get_rank()} to {dest}..."
            )

            # Send header
            dist_comm.send(
                (self.sub_tree_idx, self.bias_list, self.pred_param_dict_list, len(self.c_list)),
                dest=dest,
                tag=self.sub_tree_idx,
            )
            # Send Cs and then Ws
            for idx, mat in enumerate(self.c_list + self.w_list):
                dist_comm.send(mat, dest=dest, tag=self.sub_tree_idx + (idx + 1) * self.TAG_OFFSET)

            LOGGER.debug(
                f"Done sending sub-tree model {self.sub_tree_idx} from node {dist_comm.get_rank()} to {dest}."
            )

        @classmethod
        def recv(cls, dist_comm, source, recv_sub_tree_idx):
            """Receive class atrributes from source on current rank.

            Parameters:
                dist_comm (DistComm): Distributed communicator
                source (int): rank for source.
                recv_sub_tree_idx (int or np.int64): Index for sub-tree model to receive.

            Returns:
                cls
            """
            LOGGER.debug(
                f"Starts receiving sub-tree model {recv_sub_tree_idx} from source {source} for rank {dist_comm.get_rank()}..."
            )

            # Receive header
            sub_tree_idx, bias_list, pred_param_dict_list, n_depth = dist_comm.recv(
                source=source, tag=recv_sub_tree_idx
            )
            assert (
                recv_sub_tree_idx == sub_tree_idx
            ), f"{recv_sub_tree_idx} is not equal to received {sub_tree_idx}"
            # Receive Cs and Ws
            c_list = [
                dist_comm.recv(source=source, tag=recv_sub_tree_idx + (idx + 1) * cls.TAG_OFFSET)
                for idx in range(n_depth)
            ]
            w_list = [
                dist_comm.recv(
                    source=source, tag=recv_sub_tree_idx + (idx + 1 + n_depth) * cls.TAG_OFFSET
                )
                for idx in range(n_depth)
            ]

            LOGGER.debug(
                f"Done receiving sub-tree model {sub_tree_idx} from source {source} for rank {dist_comm.get_rank()}."
            )

            return cls(sub_tree_idx, c_list, w_list, bias_list, pred_param_dict_list)

        def to_xlinear_model(self):
            """Create XLinearModel from the model chains and parameters

            Returns:
                XLinearModel
            """
            model_chain = []
            for c, w, bias, pred_param_dict in zip(
                self.c_list, self.w_list, self.bias_list, self.pred_param_dict_list
            ):
                model_chain.append(MLModel(C=c, W=w, bias=bias, pred_params=pred_param_dict))

            hlm_pred_params = HierarchicalMLModel.PredParams(
                model_chain=[model.get_pred_params() for model in model_chain]
            )

            return XLinearModel(HierarchicalMLModel(model_chain, pred_params=hlm_pred_params))

        @classmethod
        def from_xlinear_model(cls, sub_tree_idx, xlinear_model):
            """Create cls from XLinearModel

            Parameters:
                sub_tree_idx (int or np.int64): The index of the sub-tree for the model.
                xlinear_model (XLinearModel): The XLinearModel for the model.

            Returns:
                cls
            """
            assert isinstance(xlinear_model, XLinearModel), type(xlinear_model)
            assert isinstance(sub_tree_idx, (int, np.int64)), type(sub_tree_idx)

            c_list = [model.C for model in xlinear_model.model.model_chain]
            w_list = [model.W for model in xlinear_model.model.model_chain]
            bias_list = [model.bias for model in xlinear_model.model.model_chain]
            pred_param_dict_list = [
                model.pred_params.to_dict() for model in xlinear_model.model.model_chain
            ]

            return cls(sub_tree_idx, c_list, w_list, bias_list, pred_param_dict_list)

    def __init__(self, dist_comm, dist_cluster_chain, train_params, pred_params, dist_params):
        assert isinstance(dist_comm, DistComm), type(dist_comm)
        assert isinstance(dist_cluster_chain, DistClusterChain), type(dist_cluster_chain)
        self._dist_comm = dist_comm
        self._train_params = train_params
        self._pred_params = pred_params
        self._dist_params = dist_params

        # Re-split for training
        self._dist_cluster_chain = dist_cluster_chain.new_instance_re_split(
            dist_params.min_n_sub_tree
        )

    def _train_meta_model(self, X, Y):
        """Train meta model."""
        LOGGER.info(
            f"Rank {self._dist_comm.get_rank()} starts meta-tree training..."
            f" {MemInfo.mem_info()}"
        )

        meta_cluster_top_chain = self._dist_cluster_chain.get_meta_tree_chain()
        meta_model = XLinearModel.train(
            X,
            self._dist_cluster_chain.get_meta_Y(Y, self._dist_params.threads),
            meta_cluster_top_chain,
            threads=self._dist_params.threads,
            train_params=self._train_params.get_meta_or_sub_tree_param(
                split_depth=self._dist_cluster_chain.get_split_depth(), is_meta_tree=True
            ),
            pred_params=self._pred_params.get_meta_or_sub_tree_param(
                split_depth=self._dist_cluster_chain.get_split_depth(), is_meta_tree=True
            ),
        )
        LOGGER.info(
            f"Rank {self._dist_comm.get_rank()} done meta-tree training." f" {MemInfo.mem_info()}"
        )

        return meta_model

    def _train_sub_models(self, X, Y, self_job):
        """Train sub models on nodes."""

        assert isinstance(self_job, XLinearLoadBalancer.DistLoadBalancedJob)

        LOGGER.info(
            f"Rank {self._dist_comm.get_rank()} get {self_job.get_n_sub_tree()} sub-trees to train"
        )
        LOGGER.info(
            f"Rank {self._dist_comm.get_rank()} starts sub-tree training..."
            f" {MemInfo.mem_info()}"
        )
        sub_model_list = []
        # Loop each sub-tree to train
        for sub_tree_idx, sub_x_idx_arr in zip(
            self_job.sub_tree_idx_arr, self_job.sub_x_idx_arr_list
        ):
            # Extract sub X and Y
            sub_y_idx_arr = self._dist_cluster_chain.get_sub_tree_assignment(sub_tree_idx)
            if sub_x_idx_arr.size != 0:
                sub_Y = Y[:, sub_y_idx_arr]
                sub_X, sub_Y = smat_util.get_row_submatrices([X, sub_Y.tocsr()], sub_x_idx_arr)
                sub_Y.tocsc()
            else:
                sub_X = csr_matrix((1, X.shape[1]), dtype=X.dtype)
                sub_Y = csc_matrix((1, len(sub_y_idx_arr)), dtype=Y.dtype)
            LOGGER.debug(
                f"Start training sub-tree {sub_tree_idx}. Sub-X shape: {sub_X.shape}. Sub-Y shape: {sub_X.shape}"
                f" {MemInfo.mem_info()}"
            )

            # Get sub-tree
            sub_tree_chain = self._dist_cluster_chain.get_sub_tree_chain(sub_tree_idx)

            # Train sub model
            sub_model = XLinearModel.train(
                sub_X,
                sub_Y,
                sub_tree_chain,
                train_params=self._train_params.get_meta_or_sub_tree_param(
                    split_depth=self._dist_cluster_chain.get_split_depth(), is_meta_tree=False
                ),
                pred_params=self._pred_params.get_meta_or_sub_tree_param(
                    split_depth=self._dist_cluster_chain.get_split_depth(), is_meta_tree=False
                ),
            )
            sub_model_list.append(self.DistSubTreeModel.from_xlinear_model(sub_tree_idx, sub_model))
            LOGGER.debug(f"Done training sub-tree {sub_tree_idx}." f" {MemInfo.mem_info()}")

        LOGGER.info(
            f"Rank {self._dist_comm.get_rank()} total {self_job.get_n_sub_tree()} sub-tree training finished."
            f" {MemInfo.mem_info()}"
        )

        return sub_model_list

    def _recv_sub_model(self, worker_job_list, worker_recv_order_list):
        """Receive sub-tree models from worker nodes on main node"""
        assert self._dist_comm.get_rank() == 0, "Can only receive on main node."

        all_sub_model_list = []
        for worker_job, worker_rank in zip(worker_job_list, worker_recv_order_list):
            assert isinstance(worker_job, XLinearLoadBalancer.DistLoadBalancedJob), type(worker_job)
            LOGGER.info(
                f"Main node start recv {worker_job.get_n_sub_tree()} sub-tree models from rank {worker_rank}"
            )
            for sub_tree_idx in worker_job.sub_tree_idx_arr:
                sub_model = self.DistSubTreeModel.recv(
                    self._dist_comm, source=worker_rank, recv_sub_tree_idx=sub_tree_idx
                )
                all_sub_model_list.append(sub_model)
            LOGGER.info(
                f"Main node done receive {worker_job.get_n_sub_tree()} sub-tree models from rank {worker_rank}"
            )

        return all_sub_model_list

    def _send_sub_model(self, self_job, sub_model_list):
        """Send sub-tree model from worker nodes to main node"""
        assert isinstance(self_job, XLinearLoadBalancer.DistLoadBalancedJob), type(self_job)
        assert isinstance(sub_model_list, list), type(sub_model_list)
        assert (
            len(sub_model_list) == self_job.get_n_sub_tree()
        ), f"{len(sub_model_list)} and {self_job.get_n_sub_tree()} length not equal."

        LOGGER.info(
            f"Rank {self._dist_comm.get_rank()} node starts sending {self_job.get_n_sub_tree()} sub-tree models."
        )
        for sub_tree_idx, sub_model in zip(self_job.sub_tree_idx_arr, sub_model_list):
            assert isinstance(sub_model, self.DistSubTreeModel), type(sub_model)
            assert (
                sub_model.sub_tree_idx == sub_tree_idx
            ), f"{sub_model.sub_tree_idx} not equal {sub_tree_idx}"
            sub_model.send(self._dist_comm, dest=0)
        LOGGER.info(
            f"Rank {self._dist_comm.get_rank()} node done sending {self_job.get_n_sub_tree()} sub-tree models."
        )

    def _reorder_sub_model_list(self, all_sub_model_list):
        """Reorder model list according to sub-tree order"""
        reordered_sub_model_list = [None] * len(all_sub_model_list)
        for sub_model in all_sub_model_list:
            assert isinstance(sub_model, self.DistSubTreeModel)
            reordered_sub_model_list[sub_model.sub_tree_idx] = sub_model

        return reordered_sub_model_list

    def dist_train(self, X, Y):
        """Distributed train model.

        Parameters:
            X (csr_matrix(float32)): instance feature matrix of shape (nr_inst, nr_feat)
            Y (csc_matrix(float32)): label matrix of shape (nr_inst, nr_labels)

        Returns:
            model (XLinearModel)
        """
        assert isinstance(X, csr_matrix), type(X)
        assert isinstance(Y, csc_matrix), type(Y)

        # Divide sub tree training jobs in a load-balanced way on main node
        # Send recv jobs containing sub trees ids to all nodes
        if self._dist_comm.get_rank() == 0:
            load_balancer = XLinearLoadBalancer(
                self._dist_comm.get_size(),
                self._dist_params.main_workload_factor,
                self._dist_params.threads,
            )
            (
                self_job,
                worker_job_list,
                worker_recv_order_list,
            ) = load_balancer.get_load_balanced_sub_train_jobs_list(self._dist_cluster_chain, Y)

            # Checks for number of sub-trees on each worker nodes
            for worker_job in worker_job_list:
                assert worker_job.get_n_sub_tree() <= self.DistSubTreeModel.TAG_OFFSET, (
                    f"Number of sub-trees: {worker_job.get_n_sub_tree()} on one worker "
                    f"exceeded tag offset: {self.DistSubTreeModel.TAG_OFFSET}, might results in send/recv errors."
                )

            for worker_rank, worker_job in zip(worker_recv_order_list, worker_job_list):
                worker_job.send(self._dist_comm, dest=worker_rank)
        else:
            self_job = XLinearLoadBalancer.DistLoadBalancedJob.recv(self._dist_comm, source=0)

        # Train meta-tree model on main node
        if self._dist_comm.get_rank() == 0:
            meta_model = self._train_meta_model(X, Y)

        # Train sub-trees models on all nodes
        sub_model_list = self._train_sub_models(X, Y, self_job)

        # Collect trained sub-tree model
        if self._dist_comm.get_rank() == 0:
            all_sub_model_list = self._recv_sub_model(worker_job_list, worker_recv_order_list)
            all_sub_model_list += sub_model_list  # Attach main sub-tree models
        else:
            self._send_sub_model(self_job, sub_model_list)

        # Concatenate all model on main node
        model = None
        if self._dist_comm.get_rank() == 0:
            LOGGER.info(
                f"Reconstruct full model on Rank {self._dist_comm.get_rank()} node..."
                f" {MemInfo.mem_info()}"
            )
            reordered_sub_model_list = self._reorder_sub_model_list(all_sub_model_list)
            sub_xlinear_models = [
                sub_model.to_xlinear_model() for sub_model in reordered_sub_model_list
            ]
            model = XLinearModel.reconstruct_model(
                meta_model,
                sub_xlinear_models,
                Y_ids_of_child_models=self._dist_cluster_chain.get_sub_tree_assignment(),
            )
            LOGGER.info(
                f"Done reconstruct full model on Rank {self._dist_comm.get_rank()} node."
                f" {MemInfo.mem_info()}"
            )

        return model


class DistributedCPUXLinearModel(object):
    """Distributed CPU XLinear training"""

    @dc.dataclass
    class TrainParams(XLinearModel.TrainParams):
        """Training parameters of Distributed XLinearModel

        Added logic of getting parameters for meta/sub-tree training.

        TODO: only `negative_sampling_scheme`='tfn' is supported currently
        """

        def get_meta_or_sub_tree_param(self, split_depth, is_meta_tree):
            """Get training parameter for meta-tree or sub-tree.

            Because XLinearModel.TrainParams could be layered, i.e. a list of separate training params
            for each cluster chain layer. Therefore, by dividing the cluster chain into meta and sub-tree,
            the training params list should also be divided into 2 corresponding parts.

            Parameters:
                split_depth (int): The depth for splitting meta-tree and sub-tree
                is_meta_tree (boolean): Whether to get params for met-tree or sub-tree
            """
            # Every layer has the same params, no need to divide
            if (
                self.hlm_args is None
                or isinstance(self.hlm_args.model_chain, MLModel.TrainParams)  # noqa: W503
                or len(self.hlm_args.model_chain) == 1  # noqa: W503
            ):
                return self

            # Split
            full_mlm_param_tuple = self.hlm_args.model_chain
            assert (
                len(full_mlm_param_tuple) > split_depth
            ), "Meta and sub-tree params should be non-empty."
            mlm_param_tuple = (
                full_mlm_param_tuple[:split_depth]
                if is_meta_tree
                else full_mlm_param_tuple[split_depth:]
            )

            # new instance
            new_train_param = self.__class__.from_dict(self.to_dict())
            new_train_param.hlm_args = self.hlm_args.__class__(model_chain=mlm_param_tuple)

            return new_train_param

    @dc.dataclass
    class PredParams(XLinearModel.PredParams):
        """Prediction parameters of Distributed XLinearModel

        Added logic of getting parameters for meta/sub-tree training.
        """

        beam_size: int = 10

        def expanding_param_chain_with_depth(self, model_depth):
            """Expanding the HierarchicalMLModel.PredParams hlm_args into a chain with given model depth.

            Also, override the `only_topk` with `beam_size` in the first depth-1 layers.
            The last 1 layer should still be `only_topk`.

            Parameters:
                model_depth (int): Known model depth for expanding the param chain.
            """
            if self.hlm_args is None:
                self.hlm_args = HierarchicalMLModel.PredParams(
                    model_chain=tuple(MLModel.PredParams() for _ in range(model_depth))
                )

            assert isinstance(self.hlm_args, HierarchicalMLModel.PredParams), type(self.hlm_args)

            # Expanding model_chain
            if isinstance(self.hlm_args.model_chain, (list, tuple)):
                # is already a list/tuple, length should equal
                assert len(self.hlm_args) == model_depth, (len(self.hlm_args), model_depth)
            elif self.hlm_args.model_chain is None:
                self.hlm_args.model_chain = tuple(MLModel.PredParams() for _ in range(model_depth))
            elif isinstance(self.hlm_args.model_chain, MLModel.PredParams):
                mlm_param_dict = self.hlm_args.model_chain.to_dict()
                self.hlm_args.model_chain = tuple(
                    MLModel.PredParams.from_dict(mlm_param_dict) for _ in range(model_depth)
                )
            else:
                raise ValueError(
                    f"Unrecongonized hlm_args.model_chain type: {type(self.hlm_args.model_chain)}"
                )

            # Override with beam_size
            self.hlm_args.override_with_kwargs({"beam_size": self.beam_size})

        def get_meta_or_sub_tree_param(self, split_depth, is_meta_tree):
            """Get prediction parameter for meta-tree or sub-tree.

            Because XLinearModel.PredParams could be layered, i.e. a list of separate prediction params
            for each cluster chain layer. Therefore, by dividing the cluster chain into meta and sub-tree,
            the prediction params list should also be divided into 2 corresponding parts.

            Parameters:
                split_depth (int): The depth for splitting meta-tree and sub-tree
                is_meta_tree (boolean): Whether to get params for met-tree or sub-tree
            """
            # Split
            full_mlm_param_tuple = self.hlm_args.model_chain
            assert isinstance(self.hlm_args.model_chain, (list, tuple)), type(
                self.hlm_args.model_chain
            )
            assert (
                len(full_mlm_param_tuple) > split_depth
            ), "Meta and sub-tree params should be non-empty."
            mlm_param_tuple = (
                full_mlm_param_tuple[:split_depth]
                if is_meta_tree
                else full_mlm_param_tuple[split_depth:]
            )

            # new instance
            new_pred_param = self.__class__.from_dict(self.to_dict())
            new_pred_param.hlm_args = self.hlm_args.__class__(model_chain=mlm_param_tuple)

            return new_pred_param

    @dc.dataclass
    class DistParams(pecos.BaseParams):
        """Distributed parameters"""

        min_n_sub_tree: int = 16
        main_workload_factor: float = 0.3
        threads: int = -1

    @classmethod
    def train(cls, dist_comm, X, Y, cluster_params, train_params, pred_params, dist_params):
        """Distributed train XLinear Model

        Parameters:
            dist_comm (DistComm): Distributed communicator.
            X (csr_matrix(float32)): instance feature matrix of shape (nr_inst, nr_feat).
            Y (csc_matrix(float32)): label matrix of shape (nr_inst, nr_labels).
            cluster_params (DistClustering.ClusterParams): Clustering parameters.
            train_params (cls.TrainParams): Training parameters.
            pred_params (cls.PredParams): Prediction parameters.
            dist_params (cls.DistParams): Distributed parameters.
        """
        assert isinstance(dist_comm, DistComm), type(dist_comm)
        assert isinstance(X, csr_matrix), type(X)
        assert isinstance(Y, csc_matrix), type(Y)
        assert isinstance(cluster_params, DistClustering.ClusterParams), type(cluster_params)
        assert isinstance(train_params, cls.TrainParams), type(train_params)
        assert isinstance(pred_params, cls.PredParams), type(pred_params)
        assert isinstance(dist_params, cls.DistParams), type(dist_params)

        # Disable XLinearModel INFO training layer logs
        # but still print DEBUG logs if overall logger level set to DEBUG
        def disable_training_layer_log():
            logging.getLogger(pecos.xmc.base.__name__).setLevel(
                logging.DEBUG if LOGGER.isEnabledFor(logging.DEBUG) else logging.WARNING
            )

        disable_training_layer_log()

        # Distributed creating cluster chain
        dist_clustering = DistClustering(dist_comm, cluster_params)
        dist_cluster_chain = dist_clustering.dist_get_cluster_chain(X, Y)

        # Distributed training model and collect results
        pred_params.expanding_param_chain_with_depth(len(dist_cluster_chain.get_cluster_chain()))
        dist_training = DistTraining(
            dist_comm, dist_cluster_chain, train_params, pred_params, dist_params
        )
        model = dist_training.dist_train(X, Y)

        return model
