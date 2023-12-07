#  Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
#  with the License. A copy of the License is located at
#
#  http://aws.amazon.com/apache2.0/
#
#  or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
#  OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
#  and limitations under the License.
from ctypes import (
    POINTER,
    c_bool,
    c_float,
    c_uint32,
    c_char_p,
    c_void_p,
)
import os
import copy
import json
import dataclasses as dc
import numpy as np
import scipy.sparse as smat

import pecos
from pecos.core import (
    ScipyCscF32,
    ScipyCsrF32,
    ScipyDrmF32,
)
from pecos.core import clib as pecos_clib


class PairwiseANN(pecos.BaseClass):
    @dc.dataclass
    class TrainParams(pecos.BaseParams):
        """Training Parameters of PairwiseANN class

        Attributes:
            metric_type (str): distance metric type, can only be "ip" for inner product for now
        """

        metric_type: str = "ip"

    @dc.dataclass
    class PredParams(pecos.BaseParams):
        """Prediction Parameters of PairwiseANN class

        Attributes:
            batch_size (int): maximum number of (input, label) pairs te be inference on for the Searchers
            only_topk (int): maximum number of candidates (sorted by distances, nearest first) return by kNN
        """

        batch_size: int = 1024
        only_topk: int = 10

    class Searchers(object):
        def __init__(self, model, pred_params, num_searcher=1):
            self.searchers_ptr = model.fn_dict["searchers_create"](
                model.model_ptr,
                num_searcher,
            )
            self.destruct_fn = model.fn_dict["searchers_destruct"]

            # searchers also hold the memory of returned np.ndarray
            self.pred_params = pred_params
            max_nnz = pred_params.batch_size * pred_params.only_topk
            self.Imat = np.zeros(max_nnz, dtype=np.uint32)
            self.Mmat = np.zeros(max_nnz, dtype=np.uint32)
            self.Dmat = np.zeros(max_nnz, dtype=np.float32)
            self.Vmat = np.zeros(max_nnz, dtype=np.float32)

        def __del__(self):
            if self.searchers_ptr is not None:
                self.destruct_fn(self.searchers_ptr)

        def ctypes(self):
            return self.searchers_ptr

        def reset(self, reset_nnz):
            self.Imat[:reset_nnz].fill(0)
            self.Mmat[:reset_nnz].fill(0)
            self.Dmat[:reset_nnz].fill(0.0)
            self.Vmat[:reset_nnz].fill(0.0)

    def __init__(
        self, model_ptr, num_input_keys, num_label_keys, feat_dim, fn_dict, pred_params=None
    ):
        """constructor of PairwiseANN class
        Args:
            model_ptr (c_void_p): pointer to C instance pecos::ann:PairwiseANN.
            num_input_keys (int): number of input keys being indexed
            num_label_keys (int): number of label keys being indexed
            feat_dim (int): feature dimension of each input features
            fn_dict (dict): dictionary that stores the C/C++ functions to call
            pred_params (PairwiseANN.PredParams, optional): prediction parameters
        """
        self.model_ptr = model_ptr
        self.num_input_keys = num_input_keys
        self.num_label_keys = num_label_keys
        self.feat_dim = feat_dim
        self.fn_dict = fn_dict
        self.pred_params = self.PredParams.from_dict(pred_params)

    def __del__(self):
        if self.model_ptr and isinstance(self.model_ptr, c_void_p):
            self.fn_dict["destruct"](self.model_ptr)

    @property
    def data_type(self):
        return self.fn_dict["data_type"]

    @property
    def metric_type(self):
        return self.fn_dict["metric_type"]

    @staticmethod
    def create_pymat(X):
        """create PyMat wrapper given the input X matrix
        Args:
            X (nd.array, scipy.sparse.csr_matrix): database matrix to be indexed. (num_item x feat_dim).
        Returns:
            pX (ScipyDrmF32/ScipyCsrF32): python wrapper class for np.array/csr_matrix
            data_type (str): data type of X, either drm or csr
        """
        pX = None
        data_type = None
        if isinstance(X, (np.ndarray, ScipyDrmF32)):
            pX = ScipyDrmF32.init_from(X)
            data_type = "drm"
        elif isinstance(X, (smat.csr_matrix, ScipyCsrF32)):
            pX = ScipyCsrF32.init_from(X)
            data_type = "csr"
        else:
            raise ValueError("type(X)={} is NOT supported!".format(type(X)))
        return pX, data_type

    @classmethod
    def train(cls, X, Y, train_params=None, pred_params=None):
        """train and return the PairwiseANN indexer
        Args:
            X (numpy.array or smat.csr_matrix): database matrix to be indexed.
                Shape of (num_input, feat_dim)
            Y (smat.csr_matrix or smat.csc_matrix): input-to-label matrix to be indexed.
                Shape of (num_input, num_label)
            train_params (Pairwise.TrainParams, optional): instance of PairwiseANN.TrainParams
            pred_params (Pairwise.PredParams, optional): instance of PairwiseANN.PredParams
        Returns:
            PairwiseANN (pecos.ann.pairwise.PairwiseANN): the trained PairwiseANN model
        """
        train_params = cls.TrainParams.from_dict(train_params)
        pred_params = cls.PredParams.from_dict(pred_params)

        if not isinstance(Y, smat.csr_matrix) and not isinstance(Y, smat.csc_matrix):
            raise ValueError("type(Y) should be either a csr_matrix or csc_matrix.")
        Y_csc = Y.tocsc()
        pY = ScipyCscF32.init_from(Y_csc)
        pX, data_type = cls.create_pymat(X)
        fn_dict = pecos_clib.pairwise_ann_init(data_type, train_params.metric_type)
        model_ptr = fn_dict["train"](pX, pY)
        return cls(model_ptr, pY.rows, pY.cols, pX.cols, fn_dict, pred_params)

    @classmethod
    def load(cls, model_folder, lazy_load=False):
        """Load PairwiseANN model from file
        Args:
            model_folder (str): model directory from which the model is loaded.
            lazy_load (bool): whether to lazy_load memory-mapped files (default False).
        Returns:
            PairwiseANN (pecos.ann.pairwise.PairwiseANN): the loaded PairwiseANN model
        """
        with open("{}/param.json".format(model_folder), "r") as fin:
            param = json.loads(fin.read())
        if param["model"] != cls.__name__:
            raise ValueError(f"param[model] != cls.__name__")

        if not ("data_type" in param and "metric_type" in param):
            raise ValueError(f"param.json did not have data_type or metric_type!")
        fn_dict = pecos_clib.pairwise_ann_init(param["data_type"], param["metric_type"])

        c_model_dir = f"{model_folder}/c_model"
        if not os.path.isdir(c_model_dir):
            raise ValueError(f"c_model_dir did not exist: {c_model_dir}")
        model_ptr = fn_dict["load"](c_char_p(c_model_dir.encode("utf-8")), c_bool(lazy_load))
        pred_params = cls.PredParams.from_dict(param["pred_kwargs"])
        return cls(
            model_ptr,
            param["num_input_keys"],
            param["num_label_keys"],
            param["feat_dim"],
            fn_dict,
            pred_params,
        )

    def save(self, model_folder):
        """Save PairwiseANN Model to file
        Args:
            model_folder (str): model directory to which the model is saved
        """
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        param = {
            "model": self.__class__.__name__,
            "data_type": self.data_type,
            "metric_type": self.metric_type,
            "num_input_keys": self.num_input_keys,
            "num_label_keys": self.num_label_keys,
            "feat_dim": self.feat_dim,
            "pred_kwargs": self.pred_params.to_dict(),
        }
        param = self.append_meta(param)
        with open("{}/param.json".format(model_folder), "w") as fout:
            fout.write(json.dumps(param, indent=True))
        c_model_dir = f"{model_folder}/c_model"
        self.fn_dict["save"](self.model_ptr, c_char_p(c_model_dir.encode("utf-8")))

    def get_pred_params(self):
        """Return a deep copy of prediction parameters

        Returns:
            copied_pred_params (dict): Prediction parameters.
        """
        return copy.deepcopy(self.pred_params)

    def searchers_create(self, pred_params=None, num_searcher=1):
        """create searchers that pre-allocate intermediate variables (e.g., topk_queue)
        Args:
            pred_params (Pairwise.PredParams, optional): instance of pecos.ann.pairwise.Pairwise.PredParams
            num_searcher: number of searcher for multi-thread inference
        Returns:
            PairwiseANN.Searchers: the pre-allocated PairwiseANN.Searchers (class object)
        """
        if not self.model_ptr:
            raise ValueError("self.model_ptr must exist before using searchers_create()")
        if num_searcher <= 0:
            raise ValueError("num_searcher={} <= 0 is NOT valid".format(num_searcher))
        pred_params = self.get_pred_params() if pred_params is None else pred_params
        return PairwiseANN.Searchers(self, pred_params, num_searcher)

    def predict(self, input_feat, label_keys, searchers, is_same_input=False):
        """predict with multi-thread. The searchers are required to be provided.
        Args:
            input_feat (numpy.array or smat.csr_matrix): input feature matrix (first key) to find kNN.
                if is_same_input == False, the shape should be (batch_size, feat_dim).
                if is_same_input == True, the shape should be (1, feat_dim).
            label_keys (numpy.array): the label keys (second key) to find kNN.
                The shape should be (batch_size, ).
            searchers (c_void_p): pointer to C/C++ vector<pecos::ann::PairwiseANN:Searcher>.
                Created by PairwiseANN.searchers_create().
            is_same_input (bool): whether to use the same first row of X to do prediction.
                For real-time inference with same input query, set is_same_input = True.
                For batch prediction with varying input querues, set is_same_input = False.
        Returns:
            Imat (np.array): returned kNN input key indices. Shape of (batch_size, topk)
            Mmat (np.array): returned kNN masking array. 1/0 mean value IS/ISNOT presented. Shape of (batch_size, topk)
            Dmat (np.array): returned kNN distance array. Shape of (batch_size, topk)
            Vmat (np.array): returned kNN value array. Shape of (batch_size, topk)
        """
        input_feat_py, data_type = self.create_pymat(input_feat)
        if data_type != self.data_type:
            raise ValueError(
                "data_type={} is NOT consistent with self.data_type={}".format(
                    data_type, self.data_type
                )
            )
        if input_feat_py.cols != self.feat_dim:
            raise ValueError(
                "input_feat_py.cols={} is NOT consistent with self.feat_dim={}".format(
                    input_feat_py.cols, self.feat_dim
                )
            )
        if not isinstance(label_keys, np.ndarray):
            raise TypeError(f"type(label_keys) != np.array")
        if not is_same_input and input_feat_py.rows != label_keys.shape[0]:
            raise ValueError(f"input_feat_py.rows != label_keys.shape[0]")

        cur_bsz = label_keys.shape[0]
        if cur_bsz > searchers.pred_params.batch_size:
            raise ValueError(f"cur_batch_size > searchers.batch_size!")
        only_topk = searchers.pred_params.only_topk
        cur_nnz = cur_bsz * only_topk

        searchers.reset(cur_nnz)
        self.fn_dict["predict"](
            searchers.ctypes(),
            cur_bsz,
            only_topk,
            input_feat_py,
            label_keys.ctypes.data_as(POINTER(c_uint32)),
            searchers.Imat.ctypes.data_as(POINTER(c_uint32)),
            searchers.Mmat.ctypes.data_as(POINTER(c_uint32)),
            searchers.Dmat.ctypes.data_as(POINTER(c_float)),
            searchers.Vmat.ctypes.data_as(POINTER(c_float)),
            c_bool(is_same_input),
        )
        Imat = searchers.Imat[:cur_nnz].reshape(cur_bsz, only_topk)
        Mmat = searchers.Mmat[:cur_nnz].reshape(cur_bsz, only_topk)
        Dmat = searchers.Dmat[:cur_nnz].reshape(cur_bsz, only_topk)
        Vmat = searchers.Vmat[:cur_nnz].reshape(cur_bsz, only_topk)
        return Imat, Mmat, Dmat, Vmat
