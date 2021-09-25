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
    c_void_p,
)
import numpy as np
import scipy.sparse as smat
from pecos.utils import smat_util
from pecos.core import ScipyCsrF32, ScipyDrmF32
from pecos.core import clib as pecos_clib


class HNSW(object):
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

    def __init__(self, model_ptr, num_item, feat_dim, M, efC, max_level, fn_dict):
        """constructor of HNSW class
        Args:
            model_ptr (c_void_p): pointer to C instance pecos::ann:HNSW. It's obtained from HNSW.train()
            num_item (int): number of item being indexed
            feat_dim (int): feature dimension of each item
            M (int): maximum number of edges per node for HNSW graph construction at layer l=1,...,L. For layer l=0, its 2*M.
            efC (int): size of the priority queue when performing best first search during construction
            max_level (int): number of maximum layers in the hiearchical graph
            fn_dict (dict): dictionary that stores the C/C++ functions to call
        """
        self.model_ptr = model_ptr
        self.num_item = num_item
        self.feat_dim = feat_dim
        self.M = M
        self.efC = efC
        self.max_level = max_level
        self.fn_dict = fn_dict

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
    def train(cls, X, M=24, efC=100, max_level=5, metric_type="ip", threads=0):
        """train and return the ANN/HNSW indexer
        Args:
            X (nd.array/ScipyDrmF32, scipy.sparse.csr_matrix/ScipyCsrF32): database matrix to be indexed. (num_item x feat_dim).
            M (int): maximum number of edges per node for HNSW graph construction at layer l=1,...,L. For layer l=0, its 2*M.
            efC (int): size of the priority queue when performing best first search during construction
            max_level (int): number of maximum layers in the hiearchical graph
            threads (int, default 0): number of threads to use for training HNSW indexer, set to 0 to use all
        Returns:
            HNSW: the trained HNSW model (class object)
        """
        pX, data_type = cls.create_pymat(X)
        fn_dict = pecos_clib.ann_hnsw_init(data_type, metric_type)
        model_ptr = fn_dict["train"](pX, M, efC, max_level, threads)
        return cls(model_ptr, pX.rows, pX.cols, M, efC, max_level, fn_dict)

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

    def predict(self, X, efS, topk, threads=0, searchers=None, ret_csr=False):
        """predict with multi-thread. If searchers are provided, less overhead for online inference.
        Args:
            X (nd.array/ScipyDrmF32, scipy.sparse.csr_matrix/ScipyCsrF32): query matrix to be predicted. (num_query x feat_dim).
            efS (int): size of the priority queue when performing best first search during inference
            topk (int): number of maximum layers in the hiearchical graph
            threads (int): number of searcher to do inference. Overridden by numer of searchers if searchers is given.
            searchers (c_void_p): pointer to C/C++ std::vector<pecos::ann::HNSW:Searcher>. It's an object returned by self.create_searcher().
            ret_csr (bool): if true, the returns will be csr matrix. if false, return induces/distance np.array
        Returns:
            indices (np.array): returned indices array, sorted by smallest-to-largest distances. (num_query x topk)
            distances (np.array): returned dinstances array, sorted by smallest-to-largest distances (num_query x topk)
        """
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

        indices = np.zeros(pX.rows * topk, dtype=np.uint32)
        distances = np.zeros(pX.rows * topk, dtype=np.float32)
        self.fn_dict["predict"](
            self.model_ptr,
            pX,
            indices.ctypes.data_as(POINTER(c_uint32)),
            distances.ctypes.data_as(POINTER(c_float)),
            efS,
            topk,
            threads,
            None if searchers is None else searchers.ctypes(),
        )

        if not ret_csr:
            indices = indices.reshape(pX.rows, topk)
            distances = distances.reshape(pX.rows, topk)
            return indices, distances
        else:
            indptr = np.arange(0, topk * (pX.rows + 1), topk, dtype=np.uint64)
            Yp = smat_util.csr_matrix(
                (distances, indices, indptr),
                shape=(pX.rows, self.num_item),
                dtype=np.float32,
            )
            return Yp
