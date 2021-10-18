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
from ctypes import (
    POINTER,
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
from pecos.utils import smat_util
from pecos.core import ScipyCsrF32, ScipyDrmF32
from pecos.core import clib as pecos_clib


class HNSW(pecos.BaseClass):
    @dc.dataclass
    class TrainParams(pecos.BaseParams):
        """Training Parameters of HNSW class

        Attributes:
            M (int): maximum number of edges per node for layer l=1,...,L. For layer l=0, its 2*M. Default 32
            efC (int): size of the priority queue when performing best first search during construction. Default 100
            threads (int): number of threads to use for training HNSW indexer. Default -1 to use all
            max_level_upper_bound (int): number of maximum layers in the hierarchical graph. Default -1 to ignore
            metric_type (str): distance metric type, can be "ip" for inner product or "l2" for Euclidean distance
        """

        M: int = 32
        efC: int = 100
        threads: int = -1
        max_level_upper_bound: int = -1
        metric_type: str = "ip"

    @dc.dataclass
    class PredParams(pecos.BaseParams):
        """Prediction Parameters of HNSW class

        Attributes:
            efS (int): size of the priority queue when performing best first search during inference
            topk (int): maximum number of candidates (sorted by distances, nearest first) return by the searcher per query
            threads (int): number of searcher to do inference in parallel.
        """

        efS: int = 100
        topk: int = 10
        threads: int = 1

    class Searchers(object):
        def __init__(self, model, num_searcher=1):
            self.searchers_ptr = model.fn_dict["searchers_create"](
                model.model_ptr,
                num_searcher,
            )
            self.destruct_fn = model.fn_dict["searchers_destruct"]

        def __del__(self):
            if self.searchers_ptr is not None:
                self.destruct_fn(self.searchers_ptr)

        def ctypes(self):
            return self.searchers_ptr

    def __init__(self, model_ptr, num_item, feat_dim, fn_dict, pred_params=None):
        """constructor of HNSW class
        Args:
            model_ptr (c_void_p): pointer to C instance pecos::ann:HNSW. It's obtained from HNSW.train()
            num_item (int): number of item being indexed
            feat_dim (int): feature dimension of each item
            fn_dict (dict): dictionary that stores the C/C++ functions to call
            pred_params (HNSW.PredParams, optional): instance of pecos.ann.hnsw.HNSW.PredParams
        """
        self.model_ptr = model_ptr
        self.num_item = num_item
        self.feat_dim = feat_dim
        self.fn_dict = fn_dict
        self.pred_params = self.PredParams.from_dict(pred_params)

    def __del__(self):
        if self.model_ptr and type(self.model_ptr) == c_void_p:
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
    def train(cls, X, train_params=None, pred_params=None):
        """train and return the ANN/HNSW indexer
        Args:
            X (nd.array/ScipyDrmF32, scipy.sparse.csr_matrix/ScipyCsrF32): database matrix to be indexed. (num_item x feat_dim).
            train_params (HNSW.TrainParams, optional): instance of pecos.ann.hnsw.HNSW.TrainParams
            pred_params (HNSW.PredParams, optional): instance of pecos.ann.hnsw.HNSW.PredParams
        Returns:
            HNSWModel (pecos.ann.hnsw.HNSW): the trained HNSW model
        """
        train_params = cls.TrainParams.from_dict(train_params)
        pred_params = cls.PredParams.from_dict(pred_params)

        pX, data_type = cls.create_pymat(X)
        fn_dict = pecos_clib.ann_hnsw_init(data_type, train_params.metric_type)
        model_ptr = fn_dict["train"](
            pX,
            train_params.M,
            train_params.efC,
            train_params.threads,
            train_params.max_level_upper_bound,
        )
        return cls(model_ptr, pX.rows, pX.cols, fn_dict, pred_params)

    @classmethod
    def load(cls, model_folder):
        """Load HNSW model from file
        Args:
            model_folder (str): model directory from which the model is loaded.
        Returns:
            HNSWModel (pecos.ann.hnsw.HNSW): the loaded HNSW model
        """
        with open("{}/param.json".format(model_folder), "r") as fin:
            param = json.loads(fin.read())
        if param["model"] != cls.__name__:
            raise ValueError(f"param[model] != cls.__name__")

        if not ("data_type" in param and "metric_type" in param):
            raise ValueError(f"param.json did not have data_type or metric_type!")
        fn_dict = pecos_clib.ann_hnsw_init(param["data_type"], param["metric_type"])

        if "c_model" not in param:
            raise ValueError(f"param.json did not have c_model")
        model_ptr = fn_dict["load"](c_char_p(param["c_model"].encode("utf-8")))
        pred_params = cls.PredParams.from_dict(param["pred_kwargs"])
        return cls(model_ptr, param["num_item"], param["feat_dim"], fn_dict, pred_params)

    def save(self, model_folder):
        """Save HNSW Model to file
        Args:
            model_folder (str): model directory to which the model is saved
        """
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        param = {
            "model": self.__class__.__name__,
            "data_type": self.data_type,
            "metric_type": self.metric_type,
            "num_item": self.num_item,
            "feat_dim": self.feat_dim,
            "pred_kwargs": self.pred_params.to_dict(),
            "c_model": f"{model_folder}/c_model",
        }
        param = self.append_meta(param)
        with open("{}/param.json".format(model_folder), "w") as fout:
            fout.write(json.dumps(param, indent=True))
        self.fn_dict["save"](self.model_ptr, c_char_p(param["c_model"].encode("utf-8")))

    def searchers_create(self, num_searcher=1):
        """create searchers that pre-allocate intermediate variables (e.g., set of visited nodes, priority queues, etc) for HNSW graph search
        Args:
            num_searcher: number of searcher for multi-thread inference
        Returns:
            HNSW.Searchers: the pre-allocated HNSW.Searchers (class object)
        """
        if not self.model_ptr:
            raise ValueError("self.model_ptr must exist before using self.create_searcher()")
        if num_searcher <= 0:
            raise ValueError("num_searcher={} <= 0 is NOT valid".format(num_searcher))
        return HNSW.Searchers(self, num_searcher)

    def get_pred_params(self):
        """Return a deep copy of prediction parameters

        Returns:
            copied_pred_params (dict): Prediction parameters.
        """
        return copy.deepcopy(self.pred_params)

    def predict(self, X, pred_params=None, searchers=None, ret_csr=True):
        """predict with multi-thread. If searchers are provided, less overhead for online inference.
        Args:
            X (nd.array/ScipyDrmF32, scipy.sparse.csr_matrix/ScipyCsrF32): query matrix to be predicted. (num_query x feat_dim).
            pred_params (HNSW.PredParams, optional): instance of pecos.ann.hnsw.HNSW.PredParams
            searchers (c_void_p): pointer to C/C++ std::vector<pecos::ann::HNSW:Searcher>. It's an object returned by self.create_searcher().
            ret_csr (bool): if true, the returns will be csr matrix. if false, return indices/distances np.array (default true)
        Returns:
            indices (np.array): returned indices array, sorted by smallest-to-largest distances. (num_query x pred_params.topk)
            distances (np.array): returned dinstances array, sorted by smallest-to-largest distances (num_query x pred_params.topk)
        """
        pred_params = self.get_pred_params() if pred_params is None else pred_params
        pX, data_type = self.create_pymat(X)
        if data_type != self.data_type:
            raise ValueError(
                "data_type={} is NOT consistent with self.data_type={}".format(
                    data_type, self.data_type
                )
            )
        if pX.cols != self.feat_dim:
            raise ValueError(
                "pX.cols={} is NOT consistent with self.feat_dim={}".format(pX.cols, self.feat_dim)
            )

        indices = np.zeros(pX.rows * pred_params.topk, dtype=np.uint32)
        distances = np.zeros(pX.rows * pred_params.topk, dtype=np.float32)
        self.fn_dict["predict"](
            self.model_ptr,
            pX,
            indices.ctypes.data_as(POINTER(c_uint32)),
            distances.ctypes.data_as(POINTER(c_float)),
            pred_params.efS,
            pred_params.topk,
            pred_params.threads,
            None if searchers is None else searchers.ctypes(),
        )

        if not ret_csr:
            indices = indices.reshape(pX.rows, pred_params.topk)
            distances = distances.reshape(pX.rows, pred_params.topk)
            return indices, distances
        else:
            indptr = np.arange(
                0, pred_params.topk * (pX.rows + 1), pred_params.topk, dtype=np.uint64
            )
            Yp = smat_util.csr_matrix(
                (distances, indices, indptr),
                shape=(pX.rows, self.num_item),
                dtype=np.float32,
            )
            return Yp
