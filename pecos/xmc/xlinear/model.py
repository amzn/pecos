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
import os
from os import path

import dataclasses as dc
import numpy as np
import pecos
import scipy.sparse as smat
from pecos.utils import smat_util
from pecos.utils.cluster_util import ClusterChain
from pecos.xmc import HierarchicalMLModel, MLModel, MLProblem


class XLinearModel(pecos.BaseClass):
    """Linear models for extreme multi-label classification.

    See more details in Algorithm 2 of PECOS paper (Yu et al., 2020).

    PECOS: Prediction for Enormous and Correlated Output Spaces
        Hsiang-Fu Yu, Kai Zhong, Inderjit S. Dhillon
        https://arxiv.org/abs/2010.05878
    """

    @dc.dataclass
    class TrainParams(pecos.BaseParams):
        """Training parameters of XLinearModel

        Attributes:
            mode (str, optional): training mode, one of [full-model(default)| matcher| ranker]
                Examples:
                    Given X, Y and a ClusterChain C_chain with D layers.
                    (1) XLinearModel.train(X, Y, C=C_chain, mode='full-model') returns a D layer model.
                    (2) XLinearModel.train(X, Y, C=C_chain, mode='matcher', ranker_level=t) returns a matcher model trained on top (D-t) layers of C_chain
                    (3) XLinearModel.train(X, Y, C=C_chain, mode='ranker', ranker_level=t) returns a ranker model trained on bottom t layers of C_chain
                    (4) XLinearModel.train(X, Y, C=None, mode='full-model') for a single OVA model
            ranker_level (int, optional): the level above leaf node to split matcher/ranker.
                Ignored when mode='full-model'. Defaults to 1.
            nr_splits (int, optional): number of splits used to construct indexer hierarchy if C is not a complete chain.
                Ignored if shallow is True. Defaults to 2.
            min_codes (int, optional): the minimal number of clusters in the cluster chain. Defaults to nr_splits
            shallow (bool, optional): whether to continue constructing a full cluster chain based on the C given, default False
            rel_mode (bool, optional): mode to use relevance score for cost sensitive learning
                    'disable': do not use cost-sensitive learning (default)
                    'induce': induce relevance matrix into relvance chain by label aggregation.
                            Use all 1.0 if relevance score is not provided.
                    'ranker-only': only use cost-sensitive learning for ranker.
            rel_norm (str, optional): norm type to row-wise normalzie relevance matrix for cost-sensitive learning
            hlm_args (HierarchicalMLModel.TrainParams, optional): HierarchicalMLModel.TrainParams. Default None.
        """

        mode: str = "full-model"
        ranker_level: int = 1
        nr_splits: int = 16
        min_codes: int = None  # type: ignore
        shallow: bool = False
        rel_mode: str = "disable"
        rel_norm: str = "no-norm"
        hlm_args: HierarchicalMLModel.TrainParams = None  # type: ignore

    @dc.dataclass
    class PredParams(pecos.BaseParams):
        """Prediction parameters of XLinearModel

        Attributes:
            hlm_args (HierarchicalMLModel.PredParams, optional): Default None.
        """

        hlm_args: HierarchicalMLModel.PredParams = None  # type: ignore

        def override_with_kwargs(self, pred_kwargs):
            self.hlm_args.override_with_kwargs(pred_kwargs)
            return self

    def __init__(self, model=None):
        """Initialization

        Args:
            model (HierarchicalMLModel): The XLinear model.
        """
        self.model = model

    def save(self, model_folder):
        """Save the XLinear model to file

        Args:
            model_folder (str): dir to save the model
        """
        if not path.exists(model_folder):
            os.makedirs(model_folder)
        param = self.append_meta({})
        with open(f"{model_folder}/param.json", "w", encoding="utf-8") as fout:
            fout.write(json.dumps(param, indent=True))
        self.model.save(path.join(model_folder, "ranker"))

    @classmethod
    def load(cls, model_folder, is_predict_only=False, **kwargs):
        """Load the XLinear model from file

        Args:
            model_folder (str): The path to the folder that stores the model.
            is_predict_only (bool): If true, the XLinear model will be loaded in C++ and can only be used for prediction.
            If False, the model is loaded in Python and can be changed such as pruning. The predict-only mode typically
            gives better performance but also means that any method not tied to prediction (e.g., train) cannot be invoked.

            kwargs:
                weight_matrix_type (string, used when is_predict_only=True): A string determining which type of weight matrix to use.
                    The allowable types are:
                        * "BINARY_SEARCH_CHUNKED": The default, stores the weight matrix in chunked format and
                    uses binary search for vector x chunk products. Typically runs the fastest unless
                    the queries are extremely sparse.
                        * "HASH_CHUNKED": Stores the weight matrix in chunked format, but uses a hash table
                    for random row access in the chunks. This hash table is used to compute vector
                    x chunk products.
                        * "CSC": Typically the slowest option. Stories the weight matrix in csc format.
                    This format tends to be the fastest to load.

                    Note: If you intend to use this model for prediction with dense queries, weight-matrix must be csc.
        Returns:
            XLinearModel
        """
        model = HierarchicalMLModel.load(
            path.join(model_folder, "ranker"), is_predict_only, **kwargs
        )
        return cls(model)

    @property
    def is_predict_only(self):
        """
        Whether or not this model has been loaded in predict only mode.
        """
        return self.model.is_predict_only

    @property
    def nr_labels(self):
        """
        Get the number of labels
        """
        return self.model.nr_labels

    @classmethod
    def train(
        cls,
        X,
        Y,
        C=None,
        R=None,
        user_supplied_negatives=None,
        train_params=None,
        pred_params=None,
        **kwargs,
    ):
        """Training method for XLinearModel

        Args:
            X (csr_matrix(float32) or ndarray(float32)): instance feature matrix of shape (nr_inst, nr_feat)
            Y (csc_matrix(float32)): label matrix of shape (nr_inst, nr_labels)
            C (csc_matrix(float32), list/tuple of csc_matrices or ClusterChain, optional): indexer matrix or cluster chain.
                Defaults to None
            R (csc_matrix(float32)): relevance matrix for cost sensitive learning. Default None.
            user_supplied_negatives (dict, optional): dictionary of usn matching matrices.
                See ClusterChain.generate_matching_chain. Defaults to None.
            train_params (XLinearModel.TrainParams, optional): instance of XLinearModel.TrainParams
            pred_params (XLinearModel.PredParams, optional): instance of XLinearModel.PredParams
            kwargs:
                pred_kwargs (dict, optional): prediction kwargs {"beam_size": INT, "only_topk": INT, "post_processor": STR},
                    Default None to use HierarchicalMLModel.DEFAULT_PRED_KWARGS

        Returns:
            XLinearModel: the trained XLinearModel
        """

        if train_params is None:  # for backward compatibility
            train_params = cls.TrainParams.from_dict(kwargs)
            train_params.hlm_args = HierarchicalMLModel.TrainParams(
                neg_mining_chain=kwargs.get("negative_sampling_scheme", "tfn"),
                model_chain=MLModel.TrainParams.from_dict(kwargs),
            )
        else:
            train_params = cls.TrainParams.from_dict(train_params)

        if pred_params is None:
            pred_params = cls.PredParams()
            pred_params.hlm_args = HierarchicalMLModel.PredParams(model_chain=MLModel.PredParams())
        else:
            pred_params = cls.PredParams.from_dict(pred_params)
        # we don't override pred_params with kwargs["pred_kwargs"] because model depth is unknown!
        if kwargs.get("pred_kwargs", None) is None:
            kwargs["pred_kwargs"] = dict()
            for kw in ["beam_size", "only_topk", "post_processor"]:
                kwargs["pred_kwargs"][kw] = kwargs.get(kw, None)

        if not train_params.min_codes:
            train_params.min_codes = train_params.nr_splits

        if C is None or (isinstance(C, (list, tuple)) and len(C) == 0):
            clustering = None
            matching_chain = None
            relevance_chain = None
        else:
            if train_params.shallow:
                clustering = ClusterChain.from_partial_chain(C, min_codes=None)
            else:
                clustering = ClusterChain.from_partial_chain(
                    C, min_codes=train_params.min_codes, nr_splits=train_params.nr_splits
                )
            matching_chain = clustering.generate_matching_chain(user_supplied_negatives)

            if train_params.rel_mode == "disable":
                relevance_chain = [None] * len(clustering)
            elif train_params.rel_mode == "induce":
                relevance_chain = clustering.generate_relevance_chain(
                    {0: R if R is not None else smat_util.binarized(Y)},
                    norm_type=train_params.rel_norm,
                    induce=True,
                )
            elif train_params.rel_mode == "ranker-only":
                relevance_chain = clustering.generate_relevance_chain(
                    {0: R},
                    norm_type=train_params.rel_norm,
                    induce=False,
                )
            else:
                raise ValueError(f"Wrong value for rel_mode: {train_params.rel_mode}")

        if train_params.mode == "full-model":
            pass
        elif train_params.mode == "matcher":
            if clustering is None:
                raise ValueError("Expect non-trivial clustering for matcher mode")
            for cc in reversed(clustering[-train_params.ranker_level :]):
                Y = Y.dot(cc).tocsc()
            clustering = ClusterChain(clustering[: -train_params.ranker_level])
            matching_chain = matching_chain[: -train_params.ranker_level]
            relevance_chain = relevance_chain[: -train_params.ranker_level]
        elif train_params.mode == "ranker":
            if clustering is None:
                raise ValueError("Expect non-trivial clustering for ranker mode")
            clustering = ClusterChain(clustering[-train_params.ranker_level :])
            matching_chain = matching_chain[-train_params.ranker_level :]
            relevance_chain = relevance_chain[-train_params.ranker_level :]
        else:
            raise ValueError(f"Wrong value for the mode attribute: {train_params.mode}")

        # include R in prob for OVA training
        prob = MLProblem(X, Y, R=R if C is None else None)

        model = HierarchicalMLModel.train(
            prob,
            clustering=clustering,
            relevance_chain=relevance_chain,
            matching_chain=matching_chain,
            train_params=train_params.hlm_args,
            pred_params=pred_params.hlm_args,
            **kwargs,
        )
        return cls(model)

    def set_output_constraint(self, labels_to_keep):
        """
        Prune clustering tree to only output labels in labels_to_keep set.
        Update C matrices from bottom-up.

        Args:
            labels_to_keep (iterable over int): label indices to keep
        """
        self.model.set_output_constraint(labels_to_keep)

    def get_submodel_rooted_at(self, given_depth, child_node_id, reindex=False):
        """
        Get sub-model which represents the subtree of child_node_id and at depth: given_depth.

        Args:
            given_depth (int): depth of the model chain
            child_node_id (int): index of the node whose subtree needs to be fetched
            reindex (bool, optional):
                if True: shapes of C and W matrices are changed
                if False(default): the shapes do not change but the matrix is sparsified

        Returns:
            ret (tuple): tuple of,
                subtree as XLinearModel object
                mapping of indices of the final model to original labels (or None if reindex = False)
        """
        sub_model, mapping_to_labels = self.model.get_submodel_rooted_at(
            given_depth, child_node_id, reindex
        )
        return XLinearModel(sub_model), mapping_to_labels

    def split_model_at_depth(self, given_depth, reindex=False):
        """Splits model tree at a given depth to yield separate models.

        Args:
            given_depth (int): depth at which the model needs to be split.
            reindex (bool, optional):
                if True: shapes of C and W matrices are changed
                if False: the shapes do not change but the matrix is sparsified

        Returns:
            split_model(dict): a dictionary containing the following keys:
                "parent_model":
                    a XLinearModel that has model_chain of the
                    original model till depth: depth
                "child_models":
                    A list of child model tuples where each tuple has:
                        XLinearModel corresponding to the subtree at child id corresponding to list index
                        mapping to original labels from the child model output (None if reindex is False)

        Notes:
            As an example if the model tree is a complete binary tree of depth 2,
            and we call this function with depth = 1, this is the output (reindex = True)
            "parent_model": 2*1 XLinearModel having the top-layer of the original tree
            "child_models"[0]:
                model corresponding to child tree of node 0 at height 1
                list([0, 1])
            "child_models"[1]:
                model corresponding to child tree of node 1 at height 1
                list([2, 3])
        """
        split_model = self.model.split_model_at_depth(given_depth, reindex)
        split_model["parent_model"] = XLinearModel(split_model["parent_model"])
        for i in range(len(split_model["child_models"])):
            split_model["child_models"][i] = (
                XLinearModel(split_model["child_models"][i][0]),
                split_model["child_models"][i][1],
            )
        return split_model

    @classmethod
    def reconstruct_model(cls, parent_model, child_models, Y_ids_of_child_models=None):
        """Reconstruct the model with parent_model and child_models

        Args:
            parent_model: a XLinearModel that has model_chain of the
                top part of original model
            child_models: A list of child models, each child model is a XLinearModel
            Y_ids_of_child_models: A list of list of index mapping to original labels from the child model output.
                For example, Y_ids_of_child_models[i][j] is the final label index of local label j from model i

        Returns:
            model: a XLinearModel that composed by parent_model and child_models

        Notes:
            As an example if parent_model is 2*1 XLinearModel, child_models are two: 2*1 XLinearModel,
            and Y_ids_of_child_models is [[0,1], [2,3]]
            After reconstruction, we will have a model that is a complete binary tree of depth 2, where
            its first layer is same as parent models and the second layer is same as the concat of two child_models.
        """
        parent_leaf_nr = parent_model.model.model_chain[-1].C.shape[0]
        if len(child_models) != parent_leaf_nr:
            raise ValueError("child models number should be same as parent leaf number")
        child_leaf_nr = child_models[0].model.model_chain[-1].C.shape[0]
        child_depth = len(child_models[0].model.model_chain)
        for child_model in child_models:
            if child_depth != len(child_model.model.model_chain):
                raise ValueError("child models should have the same depth")
            if child_leaf_nr != child_model.model.model_chain[-1].C.shape[0]:
                raise ValueError("child models should have the same leaf number")

        if Y_ids_of_child_models is None:
            permuted_Y_label = np.arange(int(parent_leaf_nr * child_leaf_nr))
        else:
            if len(Y_ids_of_child_models) != parent_leaf_nr:
                raise ValueError("len of Y_ids_of_child_models shild be same as parent leaf number")
            for Y_ids in Y_ids_of_child_models:
                if len(Y_ids) != child_leaf_nr:
                    raise ValueError("len of Y_ids should same as child leaf number")
            permuted_Y_label = np.concatenate(Y_ids_of_child_models)
        inv_permuted_Y_label = np.empty(len(permuted_Y_label), dtype=int)
        for i, c in enumerate(permuted_Y_label):
            inv_permuted_Y_label[c] = i
        model_chain = [model for model in parent_model.model.model_chain]
        subtree_depth = len(child_models[0].model.model_chain)
        for d in range(subtree_depth):
            subtree_Ws = [xlm.model.model_chain[d].W for xlm in child_models]
            new_W = smat_util.hstack_csc(subtree_Ws)
            subtree_Cs = [xlm.model.model_chain[d].C for xlm in child_models]
            new_C = smat_util.block_diag_csc(subtree_Cs)
            # this step is to avoid inconsistant of data type in large scale data
            new_W = smat.csc_matrix((new_W.data, new_W.indices, new_W.indptr), shape=new_W.shape)
            new_C = smat.csc_matrix((new_C.data, new_C.indices, new_C.indptr), shape=new_C.shape)
            if d == subtree_depth - 1:
                new_W = new_W[:, inv_permuted_Y_label]
                new_C.indices = permuted_Y_label[new_C.indices]
            model_chain.append(
                MLModel(C=new_C, W=new_W, bias=child_models[0].model.model_chain[d].bias)
            )
        new_params = parent_model.model.get_pred_params() + child_models[0].model.get_pred_params()
        return cls(
            HierarchicalMLModel(
                model_chain,
                pred_params=new_params,
            )
        )

    @staticmethod
    def save_feature_matrix(tgt, feat_mat):
        """Save feature matrix to file

        Args:
            tgt (str or file-like object): destination to save the feature matrix
            feat_mat (sparse matrix or ndarray): feature matrix to save
        """
        smat_util.save_matrix(tgt, feat_mat)

    @staticmethod
    def load_feature_matrix(src):
        """Load feature matrix from file

        Args:
            src (str or file-like object): file to load the feature matrix

        Returns:
            matrix (csr_matrix or ndarray): loaded feature matrix
        """
        feat_mat = smat_util.load_matrix(src)
        if isinstance(feat_mat, np.ndarray):
            feat_mat = np.ascontiguousarray(feat_mat)
        elif isinstance(feat_mat, smat.spmatrix):
            feat_mat = feat_mat.tocsr()
            feat_mat.sort_indices()
        return feat_mat

    @staticmethod
    def load_label_matrix(src, for_training=False):
        """Load label matrix from file

        Args:
            src (str or file-like object): file to load the label matrix
            for_training (bool, optional): if False(default) return csr_matrix, else return csc_matrix

        Returns:
            matrix (csr_matrix or csc_matrix): loaded label matrix
        """
        assert isinstance(src, str), "src for load_label_matrix must be a str"
        dtype = np.float32
        feat_mat = smat_util.load_matrix(src)
        feat_mat = feat_mat.tocsc() if for_training else feat_mat.tocsr()
        return feat_mat.astype(dtype)

    def get_pred_params(self):
        """Get HierarchicalMLModel's pred_params for creating the XLinearModel.PredParams instance

        Returns:
            PredParams: Model's prediction parameters.
        """
        ret_pred_params = self.PredParams(
            hlm_args=self.model.get_pred_params(),
        )
        return ret_pred_params

    def predict(
        self,
        X,
        pred_params=None,
        selected_outputs_csr=None,
        **kwargs,
    ):
        """Predict on given input data

        Args:
            X (csr_matrix(float32) or ndarray(float32)): instance feature matrix to predict on
            pred_params (XLinearModel.PredParams, optional): instance of XLinearModel.PredParams
            selected_outputs_csr (csr_matrix, optional): instance label matrix to predict from with
                shape (instances, labels). Interested labels to predict are denoted only by indices
                with a nonzero value.
            kwargs:
                beam_size (int, optional): override the beam size specified in the model.
                    Default None to disable overriding
                only_topk (int, optional): override the only topk specified in the model
                    Default None to disable overriding
                post_processor (str, optional):  override the post_processor specified in the model
                    Default None to disable overriding
                threads (int, optional): the number of threads to use for prediction.
                    Defaults to -1 to use all
                max_pred_chunk (int, optional): the number of instances to predict in a batch. Set to None to ignore. Default 10^7

        Returns:
            Y_pred (csr_matrix): If a selected output matrix is provided, the prediction for the
                indicated output is returned. Otherwise, a prediction matrix for some topk
                scores is returned.
        """
        if (pred_params is not None) and (not isinstance(pred_params, self.PredParams)):
            raise TypeError("type(pred_kwargs) is not supported")

        max_pred_chunk = kwargs.get("max_pred_chunk", 10 ** 7)
        if not isinstance(max_pred_chunk, (int, None)):
            raise TypeError("type(max_pred_chunk) is not supported.")

        if max_pred_chunk is None or max_pred_chunk >= X.shape[0]:
            if selected_outputs_csr is None:
                Y_pred = self.model.predict(
                    X,
                    pred_params=None if pred_params is None else pred_params.hlm_args,
                    **kwargs,
                )
            else:
                Y_pred = self.model.predict_on_selected_outputs(
                    X,
                    selected_outputs_csr,
                    pred_params=None if pred_params is None else pred_params.hlm_args,
                    **kwargs,
                )
        else:
            Ys = []
            new_kwargs = kwargs.copy()
            new_kwargs.pop("max_pred_chunk", None)
            for i in range(0, X.shape[0], max_pred_chunk):
                Ye = self.predict(
                    X[i : i + max_pred_chunk, :],
                    pred_params=pred_params,
                    selected_outputs_csr=selected_outputs_csr[i : i + max_pred_chunk, :]
                    if selected_outputs_csr is not None
                    else None,
                    **new_kwargs,
                )
                Ys.append(Ye)
            Y_pred = smat_util.vstack_csr(Ys)

        return Y_pred
