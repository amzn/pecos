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
import copy
import ctypes
import logging
import os
from ctypes import (
    CDLL,
    CFUNCTYPE,
    POINTER,
    byref,
    c_bool,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_int32,
    c_uint32,
    c_uint64,
    c_void_p,
    cast,
)
from glob import glob
from subprocess import check_output

import numpy as np
import pecos
import scipy.sparse as smat
from pecos.utils import smat_util

LOGGER = logging.getLogger("__name__")

XLINEAR_SOLVERS = {
    "L2R_L2LOSS_SVC_DUAL": 1,
    "L2R_L1LOSS_SVC_DUAL": 3,
    "L2R_LR_DUAL": 7,
    "L2R_L2LOSS_SVC_PRIMAL": 2,
}
# Ordering must be consistent with with layer_type_t definition within inference.hpp
XLINEAR_INFERENCE_MODEL_TYPES = {"CSC": 0, "HASH_CHUNKED": 1, "BINARY_SEARCH_CHUNKED": 2}
TFIDF_TOKENIZER_CODES = {"word": 10, "char": 20, "char_wb": 30}


class TfidfBaseVectorizerParam(ctypes.Structure):
    """
    python class for handling struct TfidfBaseVectorizerParam in tfidf.hpp
    """

    _fields_ = [
        ("min_ngram", c_int32),
        ("max_ngram", c_int32),
        ("max_length", c_int32),
        ("max_feature", c_int32),
        ("min_df_ratio", c_float),
        ("max_df_ratio", c_float),
        ("min_df_cnt", c_int32),
        ("max_df_cnt", c_int32),
        ("binary", c_bool),
        ("use_idf", c_bool),
        ("smooth_idf", c_bool),
        ("add_one_idf", c_bool),
        ("sublinear_tf", c_bool),
        ("keep_frequent_feature", c_bool),
        ("norm_p", c_int32),
        ("tok_type", c_int32),
    ]

    DEFAULTS = {
        "min_ngram": 1,
        "max_ngram": 1,
        "max_length": -1,
        "max_feature": 0,
        "min_df_ratio": 0.0,
        "max_df_ratio": 1.0,
        "min_df_cnt": 0,
        "max_df_cnt": -1,
        "binary": False,
        "use_idf": True,
        "smooth_idf": True,
        "add_one_idf": False,
        "sublinear_tf": False,
        "keep_frequent_feature": True,
        "norm_p": 2,
        "tok_type": TFIDF_TOKENIZER_CODES["word"],
    }

    @classmethod
    def get_default(cls, name):
        return copy.deepcopy(cls.DEFAULTS[name])

    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = {}

        def extract_dict_key(config_dict, key, alias):
            return config_dict.get(key, config_dict.get(alias, self.get_default(key)))

        config_dict["norm_p"] = extract_dict_key(config_dict, "norm_p", "norm")
        # to support norm_p being "l1" or "l2"
        if isinstance(config_dict["norm_p"], str):
            config_dict["norm_p"] = int(config_dict["norm_p"][1:])
        if not (config_dict["norm_p"] == 1 or config_dict["norm_p"] == 2):
            raise NotImplementedError("norm_p only support 1 or 2")

        config_dict["tok_type"] = extract_dict_key(config_dict, "tok_type", "analyzer")
        if isinstance(config_dict["tok_type"], str):
            config_dict["tok_type"] = TFIDF_TOKENIZER_CODES[config_dict["tok_type"]]

        config_dict["max_length"] = extract_dict_key(config_dict, "max_length", "truncate_length")

        if "ngram_range" in config_dict:
            config_dict["min_ngram"] = config_dict["ngram_range"][0]
            config_dict["max_ngram"] = config_dict["ngram_range"][1]

        name2type = dict(TfidfBaseVectorizerParam._fields_)
        for name in name2type:
            setattr(self, name, name2type[name](config_dict.get(name, self.get_default(name))))


class TfidfVectorizerParam(ctypes.Structure):
    """
    python class for handling struct TfidfVectorizerParam in tfidf.hpp
    """

    _fields_ = [
        ("base_param_ptr", POINTER(TfidfBaseVectorizerParam)),
        ("num_base_vect", c_int32),
        ("norm_p", c_int32),
    ]

    def __init__(self, base_vect_param_list, norm_p):

        self.num_base_vect = len(base_vect_param_list)
        self.c_base_params = (TfidfBaseVectorizerParam * self.num_base_vect)()
        for i, base_vect_param in enumerate(base_vect_param_list):
            self.c_base_params[i] = base_vect_param

        self.base_param_ptr = cast(self.c_base_params, POINTER(TfidfBaseVectorizerParam))
        self.num_base_vect = c_int32(self.num_base_vect)
        self.norm_p = c_int32(norm_p)


class ScipyCscF32(ctypes.Structure):
    """
    PyMatrix for scipy.sparse.csc_matrix
    """

    _fields_ = [
        ("rows", c_uint32),
        ("cols", c_uint32),
        ("col_ptr", POINTER(c_uint64)),
        ("row_idx", POINTER(c_uint32)),
        ("val", POINTER(c_float)),
    ]

    def __init__(self, A):
        assert isinstance(A, smat.csc_matrix)
        assert A.dtype == np.float32
        self.py_buf = {
            "col_ptr": A.indptr.astype(np.uint64, copy=False),
            "row_idx": A.indices.astype(np.uint32, copy=False),
            "val": A.data.astype(np.float32, copy=False),
        }

        self.rows = c_uint32(A.shape[0])
        self.cols = c_uint32(A.shape[1])
        name2type = dict(ScipyCscF32._fields_)
        for name in self.py_buf:
            setattr(self, name, self.py_buf[name].ctypes.data_as(name2type[name]))
        self.buf = A

    @property
    def dtype(self):
        return self.buf.dtype

    @property
    def shape(self):
        return self.buf.shape

    @classmethod
    def init_from(cls, A):
        if A is None:
            return None
        elif isinstance(A, cls):
            return A
        else:
            return cls(A)


class ScipyCsrF32(ctypes.Structure):
    """
    PyMatrix for scipy.sparse.csr_matrix
    """

    _fields_ = [
        ("rows", c_uint32),
        ("cols", c_uint32),
        ("row_ptr", POINTER(c_uint64)),
        ("col_idx", POINTER(c_uint32)),
        ("val", POINTER(c_float)),
    ]

    def __init__(self, A):
        assert isinstance(A, smat.csr_matrix)
        assert A.dtype == np.float32
        self.py_buf = {
            "row_ptr": A.indptr.astype(np.uint64, copy=False),
            "col_idx": A.indices.astype(np.uint32, copy=False),
            "val": A.data.astype(np.float32, copy=False),
        }

        self.rows = c_uint32(A.shape[0])
        self.cols = c_uint32(A.shape[1])
        name2type = dict(ScipyCsrF32._fields_)
        for name in self.py_buf:
            setattr(self, name, self.py_buf[name].ctypes.data_as(name2type[name]))
        self.buf = A

    @classmethod
    def init_from(cls, A):
        if A is None:
            return None
        elif isinstance(A, cls):
            return A
        else:
            return cls(A)

    @property
    def dtype(self):
        return self.buf.dtype

    @property
    def shape(self):
        return self.buf.shape

    def dot(self, other):
        return self.buf.dot(other)


class ScipyDrmF32(ctypes.Structure):
    """
    PyMatrix for row-major scipy.ndarray
    """

    _fields_ = [("rows", c_uint32), ("cols", c_uint32), ("val", POINTER(c_float))]

    def __init__(self, A):
        assert isinstance(A, np.ndarray)
        assert A.dtype == np.float32
        assert A.flags.c_contiguous is True
        self.py_buf = {"val": A}

        self.rows = c_uint32(A.shape[0])
        self.cols = c_uint32(A.shape[1])
        name2type = dict(ScipyDrmF32._fields_)
        for name in self.py_buf:
            setattr(self, name, self.py_buf[name].ctypes.data_as(name2type[name]))
        self.buf = A

    @classmethod
    def init_from(cls, A):
        if A is None:
            return None
        elif isinstance(A, cls):
            return A
        else:
            return cls(A)

    @property
    def dtype(self):
        return self.buf.dtype

    @property
    def shape(self):
        return self.buf.shape

    def dot(self, other):
        if isinstance(other, smat.spmatrix):
            return other.T.dot(self.buf.T).T
        else:
            return self.buf.dot(other)


class ScipyDcmF32(ctypes.Structure):
    """
    PyMatrix for col-major scipy.ndarray
    """

    _fields_ = [("rows", c_uint32), ("cols", c_uint32), ("val", POINTER(c_float))]

    def __init__(self, A):
        assert isinstance(A, np.ndarray)
        assert A.dtype == np.float32
        assert A.flags.f_contiguous is True
        self.py_buf = {"val": A}

        self.rows = c_uint32(A.shape[0])
        self.cols = c_uint32(A.shape[1])
        name2type = dict(ScipyDcmF32._fields_)
        for name in self.py_buf:
            setattr(self, name, self.py_buf[name].ctypes.data_as(name2type[name]))
        self.buf = A

    @classmethod
    def init_from(cls, A):
        if A is None:
            return None
        elif isinstance(A, cls):
            return A
        else:
            return cls(A)

    @property
    def dtype(self):
        return self.buf.dtype

    @property
    def shape(self):
        return self.buf.shape

    def dot(self, other):
        if isinstance(other, smat.spmatrix):
            return other.T.dot(self.buf.T).T
        else:
            return self.buf.dot(other)


class ScipyCoordinateSparseAllocator(object):
    """
    Scipy Coordinate Sparse Matrix Allocator for C++/C code
    """

    CFUNCTYPE = CFUNCTYPE(None, c_uint32, c_uint32, c_uint64, c_void_p, c_void_p, c_void_p)

    def __init__(self, rows=0, cols=0, dtype=np.float64):
        self.rows = rows
        self.cols = cols
        self.row_idx = None
        self.col_idx = None
        self.data = None
        self.dtype = dtype
        assert dtype == np.float32 or dtype == np.float64

    def __call__(self, rows, cols, nnz, row_ptr, col_ptr, val_ptr):
        self.rows = rows
        self.cols = cols
        self.row_idx = np.zeros(nnz, dtype=np.uint64)
        self.col_idx = np.zeros(nnz, dtype=np.uint64)
        self.data = np.zeros(nnz, dtype=self.dtype)
        cast(row_ptr, POINTER(c_uint64)).contents.value = self.row_idx.ctypes.data_as(
            c_void_p
        ).value
        cast(col_ptr, POINTER(c_uint64)).contents.value = self.col_idx.ctypes.data_as(
            c_void_p
        ).value
        cast(val_ptr, POINTER(c_uint64)).contents.value = self.data.ctypes.data_as(c_void_p).value

    def tocoo(self):
        return smat.coo_matrix(
            (self.data, (self.row_idx, self.col_idx)), shape=(self.rows, self.cols)
        )

    def tocsr(self):
        return smat.csr_matrix(
            (self.data, (self.row_idx, self.col_idx)), shape=(self.rows, self.cols)
        )

    def tocsc(self):
        return smat.csc_matrix(
            (self.data, (self.row_idx, self.col_idx)), shape=(self.rows, self.cols)
        )

    @property
    def cfunc(self):
        return self.CFUNCTYPE(self)


class ScipyCompressedSparseAllocator(object):
    """
    Scipy Compressed Sparse Matrix Allocator for C++/C code,
    which supports both smat.csr_matrix and smat.csc_matrix.

    Whether it is row or column major is controlled by self.is_col_major,
    which is passed in by the first argument in the __call__().

    Attributes:
        CFUNCTYPE (ctypes.CFUNCTYPE): a function prototype creates functions that uses the standard C calling convention
    """

    CFUNCTYPE = CFUNCTYPE(None, c_bool, c_uint64, c_uint64, c_uint64, c_void_p, c_void_p, c_void_p)

    def __init__(self, rows=0, cols=0, dtype=np.float32):
        self.cols = cols
        self.rows = rows
        self.indices = None
        self.indptr = None
        self.data = None
        self.dtype = dtype
        self.is_col_major = None
        assert dtype == np.float32

    def __call__(self, is_col_major, rows, cols, nnz, indices_ptr, indptr_ptr, data_ptr):
        """
        Allocate memory for the members

        Parameters:
            is_col_major (bool):  specifying whether the to-be allocated matrix is row-majored or col-majored.
            rows (int): the number of rows of the sparse matrix.
            cols (int): the number of cols of the sparse matrix.
            nnz (int): the number of non-zeros of the sparse matrix.
            indptr_ptr (pointer): the pointer to the nnz array, of length (rows+1) or (cols+1).
            indices_ptr (pointer): the pointer to the row/col indices array, of length nnz.
            data_ptr (pointer): the pointer to the non-zero values array, of length nnz.

        Returns:
            None
        """

        self.cols = cols
        self.rows = rows
        self.is_col_major = is_col_major
        if is_col_major:
            self.indptr = np.zeros(cols + 1, dtype=np.uint64)
        else:
            self.indptr = np.zeros(rows + 1, dtype=np.uint64)
        self.indices = np.zeros(nnz, dtype=np.uint32)
        self.data = np.zeros(nnz, dtype=self.dtype)

        cast(indices_ptr, POINTER(c_uint64)).contents.value = self.indices.ctypes.data_as(
            c_void_p
        ).value
        cast(indptr_ptr, POINTER(c_uint64)).contents.value = self.indptr.ctypes.data_as(
            c_void_p
        ).value
        cast(data_ptr, POINTER(c_uint64)).contents.value = self.data.ctypes.data_as(c_void_p).value

    def get(self):
        if self.is_col_major:
            return smat_util.csc_matrix(
                (self.data, self.indices, self.indptr), shape=(self.rows, self.cols)
            )
        else:
            return smat_util.csr_matrix(
                (self.data, self.indices, self.indptr), shape=(self.rows, self.cols)
            )

    @property
    def cfunc(self):
        return self.CFUNCTYPE(self)


class corelib(object):
    """
    The core functions for linear problems
    """

    @staticmethod
    def fillprototype(f, restype, argtypes):
        """
        Specify corelib function's return type and argument types.

        Args:
            restype (single or list of ctypes): The return type.
            argtypes (list of ctypes): The argument types.
        """
        f.restype = restype
        f.argtypes = argtypes

    @staticmethod
    def load_dynamic_library(dirname, soname, forced_rebuild=False):
        """
        Load compiled C library into Python.
        If not found, will build upon loading.

        Args:
            dirname (str): The directory of C library.
            soname (str): The name of C library.
            force_rebuild (bool, optional): Whether to force rebuild C library upon calling.

        Return:
            c_lib (CDLL): Ctypes CDLL library.
        """
        try:
            if forced_rebuild:
                check_output("make -C {} clean lib".format(dirname), shell=True)
            path_to_so = glob(os.path.join(dirname, soname) + "*.so")[0]
            _c_lib = CDLL(path_to_so)
        except BaseException:
            try:
                check_output("make -C {} clean lib".format(dirname), shell=True)
                path_to_so = glob(os.path.join(dirname, soname) + "*.so")[0]
                _c_lib = CDLL(path_to_so)
            except BaseException:
                raise Exception("{soname} library cannot be found and built.".format(soname=soname))
        return _c_lib

    def __init__(self, dirname, soname, forced_rebuild=False):
        self.clib_float32 = corelib.load_dynamic_library(
            dirname, soname + "_float32", forced_rebuild=forced_rebuild
        )
        self.link_xlinear_methods()
        self.link_sparse_operations()
        self.link_clustering()
        self.link_tfidf_vectorizer()
        self.link_ann_hnsw_methods()

    def link_xlinear_methods(self):
        """
        Specify C-lib's Xlinear methods argument and return type.
        """
        arg_list = [
            POINTER(ScipyCsrF32),  # CSR X
            POINTER(ScipyCscF32),  # CSC Y
            POINTER(ScipyCscF32),  # CSC C
            POINTER(ScipyCscF32),  # CSC M
            POINTER(ScipyCscF32),  # CSC R
            ScipyCoordinateSparseAllocator.CFUNCTYPE,  # py_coo_allocator
            c_double,  # threshold
            c_uint32,  # max_nonzeros_per_label
            c_int,  # solver_type
            c_double,  # Cp
            c_double,  # Cn
            c_uint64,  # max_iter
            c_double,  # eps
            c_double,  # bias
            c_int,  # threads
        ]
        corelib.fillprototype(
            self.clib_float32.c_xlinear_single_layer_train_csr_f32,
            None,
            [POINTER(ScipyCsrF32)] + arg_list[1:],
        )
        corelib.fillprototype(
            self.clib_float32.c_xlinear_single_layer_train_drm_f32,
            None,
            [POINTER(ScipyDrmF32)] + arg_list[1:],
        )

        arg_list = [c_void_p]
        corelib.fillprototype(self.clib_float32.c_xlinear_destruct_model, None, arg_list)

        # Interface for sparse prediction
        arg_list = [
            c_void_p,
            POINTER(ScipyCsrF32),
            c_uint32,
            c_char_p,
            c_uint32,
            c_int,
            ScipyCompressedSparseAllocator.CFUNCTYPE,
        ]
        corelib.fillprototype(self.clib_float32.c_xlinear_predict_csr_f32, None, arg_list)

        # Interface for dense prediction
        arg_list = [
            c_void_p,
            POINTER(ScipyDrmF32),
            c_uint32,
            c_char_p,
            c_uint32,
            c_int,
            ScipyCompressedSparseAllocator.CFUNCTYPE,
        ]
        corelib.fillprototype(self.clib_float32.c_xlinear_predict_drm_f32, None, arg_list)

        # Interface for sparse selected output prediction
        arg_list = [
            c_void_p,
            POINTER(ScipyCsrF32),
            POINTER(ScipyCsrF32),
            c_char_p,
            c_int,
            ScipyCompressedSparseAllocator.CFUNCTYPE,
        ]
        corelib.fillprototype(
            self.clib_float32.c_xlinear_predict_on_selected_outputs_csr_f32, None, arg_list
        )

        # Interface for dense selected output prediction
        arg_list = [
            c_void_p,
            POINTER(ScipyDrmF32),
            POINTER(ScipyCsrF32),
            c_char_p,
            c_int,
            ScipyCompressedSparseAllocator.CFUNCTYPE,
        ]
        corelib.fillprototype(
            self.clib_float32.c_xlinear_predict_on_selected_outputs_drm_f32, None, arg_list
        )

        # c interface for loading just model tree directly (no tfidf)
        res_list = c_void_p
        arg_list = [c_char_p]
        corelib.fillprototype(self.clib_float32.c_xlinear_load_model_from_disk, res_list, arg_list)

        res_list = c_void_p
        arg_list = [c_char_p, c_int]
        corelib.fillprototype(
            self.clib_float32.c_xlinear_load_model_from_disk_ext, res_list, arg_list
        )

        # c interface for per-layer prediction
        arg_list = [
            POINTER(ScipyCsrF32),
            POINTER(ScipyCsrF32),
            POINTER(ScipyCscF32),
            POINTER(ScipyCscF32),
            c_char_p,
            c_uint32,
            c_int,
            c_float,
            ScipyCompressedSparseAllocator.CFUNCTYPE,
        ]
        corelib.fillprototype(
            self.clib_float32.c_xlinear_single_layer_predict_csr_f32, None, arg_list
        )

        arg_list = [
            POINTER(ScipyDrmF32),
            POINTER(ScipyCsrF32),
            POINTER(ScipyCscF32),
            POINTER(ScipyCscF32),
            c_char_p,
            c_uint32,
            c_int,
            c_float,
            ScipyCompressedSparseAllocator.CFUNCTYPE,
        ]
        corelib.fillprototype(
            self.clib_float32.c_xlinear_single_layer_predict_drm_f32, None, arg_list
        )

        # c interface for per-layer selected output prediction
        arg_list = [
            POINTER(ScipyCsrF32),
            POINTER(ScipyCsrF32),
            POINTER(ScipyCsrF32),
            POINTER(ScipyCscF32),
            POINTER(ScipyCscF32),
            c_char_p,
            c_int,
            c_float,
            ScipyCompressedSparseAllocator.CFUNCTYPE,
        ]
        corelib.fillprototype(
            self.clib_float32.c_xlinear_single_layer_predict_on_selected_outputs_csr_f32,
            None,
            arg_list,
        )

        arg_list = [
            POINTER(ScipyDrmF32),
            POINTER(ScipyCsrF32),
            POINTER(ScipyCsrF32),
            POINTER(ScipyCscF32),
            POINTER(ScipyCscF32),
            c_char_p,
            c_int,
            c_float,
            ScipyCompressedSparseAllocator.CFUNCTYPE,
        ]
        corelib.fillprototype(
            self.clib_float32.c_xlinear_single_layer_predict_on_selected_outputs_drm_f32,
            None,
            arg_list,
        )

        res_list = c_uint32
        arg_list = [c_void_p, c_char_p]
        corelib.fillprototype(self.clib_float32.c_xlinear_get_int_attr, res_list, arg_list)

        res_list = c_int
        arg_list = [c_void_p, c_int]
        corelib.fillprototype(self.clib_float32.c_xlinear_get_layer_type, res_list, arg_list)

    def xlinear_load_predict_only(
        self,
        folder,
        weight_matrix_type="BINARY_SEARCH_CHUNKED",
    ):
        """
        Load xlinear model in predict only mode.

        Args:
            folder (str): The folder path for xlinear model.
            weight_matrix_type (str, optional): The xlinear inference model types.

        Return:
            cmodel (ptr): The pointer to xlinear model.
        """
        weight_matrix_type_id = XLINEAR_INFERENCE_MODEL_TYPES[weight_matrix_type]
        cmodel = self.clib_float32.c_xlinear_load_model_from_disk_ext(
            c_char_p(folder.encode("utf-8")), c_int(int(weight_matrix_type_id))
        )
        return cmodel

    def xlinear_destruct_model(self, c_model):
        """
        Destruct xlinear model.

        Args:
            cmodel (ptr): The pointer to xlinear model.
        """
        self.clib_float32.c_xlinear_destruct_model(c_model)

    def xlinear_predict(
        self,
        c_model,
        X,
        overriden_beam_size,
        overriden_post_processor_str,
        overriden_only_topk,
        threads,
        pred_alloc,
    ):
        """
        Performs a full prediction using the given model and queries.

        Args:
            c_model (c_pointer): A C pointer to the model to use for prediction. This pointer
                is returned by the c_load_xlinear_model_from_disk and
                c_load_xlinear_model_from_disk_ext functions in corelib.clib_float32.
            X: The query matrix (admissible formats are smat.csr_matrix,
                np.ndarray, ScipyCsrF32, or ScipyDrmF32). Note that if this is smat.csr_matrix,
                the matrix must have sorted indices. You can call sort_indices() to ensure this.
            overriden_beam_size (uint): Overrides the beam size to use for prediction. Use None for
                model defaults.
            overriden_post_processor_str (string): Overrides the post processor to use by name. Use
                None for model defaults.
            overriden_only_topk (uint): Overrides the number of results to return for each query. Use
                None for model defaults.
            threads (int): Sets the number of threads to use in computation. Use
                -1 to use the maximum amount of available threads.
            pred_alloc (ScipyCompressedSparseAllocator): The allocator to store the result in.
        """
        clib = self.clib_float32

        if isinstance(X, smat.csr_matrix):
            if not X.has_sorted_indices:
                raise ValueError("Query matrix does not have sorted indices!")
            X = ScipyCsrF32.init_from(X)
        elif isinstance(X, np.ndarray):
            X = ScipyDrmF32.init_from(X)

        if isinstance(X, ScipyCsrF32):
            c_predict = clib.c_xlinear_predict_csr_f32
        elif isinstance(X, ScipyDrmF32):
            c_predict = clib.c_xlinear_predict_drm_f32
        else:
            raise NotImplementedError("type(X) = {} not implemented".format(type(X)))

        c_predict(
            c_model,
            byref(X),
            overriden_beam_size if overriden_beam_size else 0,
            overriden_post_processor_str.encode("utf-8") if overriden_post_processor_str else None,
            overriden_only_topk if overriden_only_topk else 0,
            threads,
            pred_alloc.cfunc,
        )

    def xlinear_predict_on_selected_outputs(
        self,
        c_model,
        X,
        selected_outputs_csr,
        overriden_post_processor_str,
        threads,
        pred_alloc,
    ):
        """
        Performs a select prediction using the given model and queries.

        Args:
            c_model (c_pointer): A C pointer to the model to use for prediction. This pointer
                is returned by the c_load_xlinear_model_from_disk and
                c_load_xlinear_model_from_disk_ext functions in corelib.clib_float32.
            X: The query matrix (admissible formats are smat.csr_matrix,
                np.ndarray, ScipyCsrF32, or ScipyDrmF32). Note that if this is smat.csr_matrix,
                the matrix must have sorted indices. You can call sort_indices() to ensure this.
            selected_outputs_csr (csr_matrix): the selected outputs to predict
            overriden_post_processor_str (string): Overrides the post processor to use by name. Use
                None for model defaults.
            threads (int): Sets the number of threads to use in computation. Use
                -1 to use the maximum amount of available threads.
            pred_alloc (ScipyCompressedSparseAllocator): The allocator to store the result in.
        """
        clib = self.clib_float32

        if isinstance(X, smat.csr_matrix):
            if not X.has_sorted_indices:
                raise ValueError("Query matrix does not have sorted indices!")
            X = ScipyCsrF32.init_from(X)
        elif isinstance(X, np.ndarray):
            X = ScipyDrmF32.init_from(X)

        if not isinstance(selected_outputs_csr, smat.csr_matrix):
            raise ValueError(
                "type(selected_outputs_csr) = {} not implemented".format(type(selected_outputs_csr))
            )
        selected_outputs_csr = ScipyCsrF32.init_from(selected_outputs_csr)

        if isinstance(X, ScipyCsrF32):
            c_predict = clib.c_xlinear_predict_on_selected_outputs_csr_f32
        elif isinstance(X, ScipyDrmF32):
            c_predict = clib.c_xlinear_predict_on_selected_outputs_drm_f32
        else:
            raise NotImplementedError("type(X) = {} not implemented".format(type(X)))

        c_predict(
            c_model,
            byref(X),
            byref(selected_outputs_csr),
            overriden_post_processor_str.encode("utf-8") if overriden_post_processor_str else None,
            threads,
            pred_alloc.cfunc,
        )

    def xlinear_single_layer_predict(
        self,
        X,
        csr_codes,
        W,
        C,
        post_processor_str,
        only_topk,
        num_threads,
        bias,
        pred_alloc,
    ):
        """
        Performs a single layer prediction in C++ using matrices owned by Python.

        Args:
            X (csr_matrix): The query matrix.
                Note that if this is smat.csr_matrix, the matrix must have sorted indices.
                You can call sort_indices() to ensure this.
            csr_codes (smat.csr_matrix or ScipyCsrF32): The prediction for the previous layer, None if this is the first layer.
            W (smat.csc_matrix, ScipyCscF32): The weight matrix for this layer.
            C (smat.csc_matrix, ScipyCscF32): The child/parent map for this layer.
            post_processor_str (str): A string specifying which post processor to use.
            only_topk (uint): How many results to return for each query.
            num_threads (uint): How many threads to use in this computation. Set to -1 to use defaults.
            bias (float): The bias of the model.
            pred_alloc (ScipyCompressedSparseAllocator): The allocator to store the result in.
        """
        clib = self.clib_float32

        post_processor_str = post_processor_str.encode("utf-8")

        W = ScipyCscF32.init_from(W)

        if isinstance(X, smat.csr_matrix):
            if not X.has_sorted_indices:
                raise ValueError("Query matrix does not have sorted indices!")
            X = ScipyCsrF32.init_from(X)
        elif isinstance(X, np.ndarray):
            X = ScipyDrmF32.init_from(X)

        if isinstance(X, ScipyCsrF32):
            c_single_layer_predict = clib.c_xlinear_single_layer_predict_csr_f32
        elif isinstance(X, ScipyDrmF32):
            c_single_layer_predict = clib.c_xlinear_single_layer_predict_drm_f32
        else:
            raise NotImplementedError("type(X) = {} not implemented".format(type(X)))

        if C is None:
            C = smat.csc_matrix(np.ones((W.shape[1], 1), dtype=W.dtype))
        C = ScipyCscF32.init_from(C)

        # csr_codes and pC might be null
        if csr_codes is not None:
            # Check that the csr_code is of valid shape
            if csr_codes.shape[0] != X.shape[0]:
                raise ValueError("Instance dimension of query and csr_codes matrix do not match")
            if csr_codes.shape[1] != C.shape[1]:
                raise ValueError("Label dimension of csr_codes and C matrix do not match")
            csr_codes = ScipyCsrF32.init_from(csr_codes)

        c_single_layer_predict(
            byref(X),
            byref(csr_codes) if csr_codes is not None else None,
            byref(W),
            byref(C),
            post_processor_str,
            only_topk,
            num_threads,
            bias,
            pred_alloc.cfunc,
        )

    def xlinear_single_layer_predict_on_selected_outputs(
        self,
        X,
        selected_outputs_csr,
        csr_codes,
        W,
        C,
        post_processor_str,
        num_threads,
        bias,
        pred_alloc,
    ):
        """
        Performs a single layer prediction in C++ using matrices owned by Python.

        Args:
            X (csr_matrix or ndarray): The query matrix.
                Note that if this is smat.csr_matrix, the matrix must have sorted indices.
                You can call sort_indices() to ensure this.
            selected_outputs_csr (csr_matrix): The selected output to predict
            csr_codes (smat.csr_matrix or ScipyCsrF32): The prediction for the previous layer, None if this is the first layer.
            W (smat.csc_matrix, ScipyCscF32): The weight matrix for this layer.
            C (smat.csc_matrix, ScipyCscF32): The child/parent map for this layer.
            post_processor_str (str): A string specifying which post processor to use.
            num_threads (uint): How many threads to use in this computation. Set to -1 to use defaults.
            bias (float): The bias of the model.
            pred_alloc (ScipyCompressedSparseAllocator): The allocator to store the result in.
        """
        clib = self.clib_float32

        post_processor_str = post_processor_str.encode("utf-8")

        W = ScipyCscF32.init_from(W)

        selected_outputs_csr = ScipyCsrF32.init_from(selected_outputs_csr)

        if isinstance(X, smat.csr_matrix):
            if not X.has_sorted_indices:
                raise ValueError("Query matrix does not have sorted indices!")
            X = ScipyCsrF32.init_from(X)
        elif isinstance(X, np.ndarray):
            X = ScipyDrmF32.init_from(X)

        if isinstance(X, ScipyCsrF32):
            c_single_layer_predict = clib.c_xlinear_single_layer_predict_on_selected_outputs_csr_f32
        elif isinstance(X, ScipyDrmF32):
            c_single_layer_predict = clib.c_xlinear_single_layer_predict_on_selected_outputs_drm_f32
        else:
            raise NotImplementedError("type(X) = {} not implemented".format(type(X)))

        if C is None:
            C = smat.csc_matrix(np.ones((W.shape[1], 1), dtype=W.dtype))
        C = ScipyCscF32.init_from(C)

        # csr_codes and pC might be null
        if csr_codes is not None:
            # Check that the csr_code is of valid shape
            if csr_codes.shape[0] != X.shape[0]:
                raise ValueError("Instance dimension of query and csr_codes matrix do not match")
            if csr_codes.shape[1] != C.shape[1]:
                raise ValueError("Label dimension of csr_codes and C matrix do not match")
            csr_codes = ScipyCsrF32.init_from(csr_codes)

        c_single_layer_predict(
            byref(X),
            byref(selected_outputs_csr),
            byref(csr_codes) if csr_codes is not None else None,
            byref(W),
            byref(C),
            post_processor_str,
            num_threads,
            bias,
            pred_alloc.cfunc,
        )

    def xlinear_single_layer_train(
        self,
        pX,
        pY,
        pC,
        pM,
        pR,
        threshold=0.1,
        max_nonzeros_per_label=None,
        solver_type="L2R_L2LOSS_SVC_DUAL",
        Cp=1.0,
        Cn=1.0,
        max_iter=1000,
        eps=0.1,
        bias=1.0,
        threads=-1,
        verbose=0,
        **kwargs,
    ):
        """
        Performs a single layer training in C++ using matrices owned by Python.

        Args:
            pX (ScipyCsrF32 or ScipyDrmF32): Instance feature matrix of shape (nr_inst, nr_feat).
            pY (ScipyCscF32): Label matrix of shape (nr_inst, nr_labels).
            pC (ScipyCscF32): Single matrix from clustering chain, representing a hierarchical clustering.
            pM (ScipyCsrF32): Single matrix from matching chain.
            pR (ScipyCscF32): Relevance matrix for cost-sensitive learning, of shape (nr_inst, nr_labels).
            threshold (float, optional): sparsify the final model by eliminating all entrees with abs value less than threshold.
                Default to 0.1.
            max_nonzeros_per_label (int, optional): keep at most NONZEROS weight parameters per label in model.
                Default None to set to (nr_feat + 1)
            solver_type (string, optional): backend linear solver type.
                Options: L2R_L2LOSS_SVC_DUAL(default), L2R_L1LOSS_SVC_DUAL.
            Cp (float, optional): positive penalty parameter. Defaults to 1.0
            Cn (float, optional): negative penalty parameter. Defaults to 1.0
            max_iter (int, optional): maximum iterations. Defaults to 100
            eps (float, optional): epsilon. Defaults to 0.1
            bias (float, optional): if >0, append the bias value to each instance feature. Defaults to 1.0
            threads (int, optional): the number of threads to use for training. Defaults to -1 to use all
            verbose (int, optional): verbose level. Defaults to 0

        Return:
            layer_train_res (smat.csc_matrix): The layer training result.
        """
        clib = self.clib_float32
        coo_alloc = ScipyCoordinateSparseAllocator(dtype=np.float32)
        if isinstance(pX, ScipyCsrF32):
            c_xlinear_single_layer_train = clib.c_xlinear_single_layer_train_csr_f32
        elif isinstance(pX, ScipyDrmF32):
            c_xlinear_single_layer_train = clib.c_xlinear_single_layer_train_drm_f32
        else:
            raise NotImplementedError("type(pX) = {} not implemented".format(type(pX)))

        c_xlinear_single_layer_train(
            byref(pX),
            byref(pY),
            byref(pC) if pC is not None else None,
            byref(pM) if pM is not None else None,
            byref(pR) if pR is not None else None,
            coo_alloc.cfunc,
            threshold,
            0 if max_nonzeros_per_label is None else max_nonzeros_per_label,
            XLINEAR_SOLVERS[solver_type],
            Cp,
            Cn,
            max_iter,
            eps,
            bias,
            threads,
        )
        return coo_alloc.tocsc().astype(np.float32)

    def xlinear_get_int_attr(self, c_model, attr):
        """
        Get int attribute from C xlinear model.

        Args:
            c_model (ptr): The C xlinear model pointer.
            attr (str): The attribute name to get.

        Return:
            int_attr (int): The int attribute under given name.
        """
        assert attr in {
            "depth",
            "nr_features",
            "nr_labels",
            "nr_codes",
        }, f"attr {attr} not implemented"
        return self.clib_float32.c_xlinear_get_int_attr(c_model, c_char_p(attr.encode("utf-8")))

    def xlinear_get_layer_type(self, c_model, layer_depth):
        """
        Get int value of layer type from a layer of a C xlinear model.

        Args:
            c_model (ptr): The C xlinear model pointer.
            layer_depth (int): The depth of the layer type to get
        """

        if layer_depth < 0 or layer_depth >= clib.xlinear_get_int_attr(c_model, "depth"):
            raise ValueError("c_model does not have a layer at depth {}".format(layer_depth))
        return self.clib_float32.c_xlinear_get_layer_type(c_model, c_int(int(layer_depth)))

    def link_sparse_operations(self):
        """
        Specify C-lib's sparse matrix operation methods argument and return type.
        """
        arg_list = [
            POINTER(ScipyCscF32),  # pX (should support both CSC and CSR)
            POINTER(ScipyCscF32),  # pY (should support both CSC and CSR)
            ScipyCompressedSparseAllocator.CFUNCTYPE,  # allocator for pZ
            c_bool,  # eliminate_zeros
            c_bool,  # sorted_indices
            c_int,  # threads
        ]
        corelib.fillprototype(
            self.clib_float32.c_sparse_matmul_csc_f32,
            None,
            [POINTER(ScipyCscF32), POINTER(ScipyCscF32)] + arg_list[2:],
        )
        corelib.fillprototype(
            self.clib_float32.c_sparse_matmul_csr_f32,
            None,
            [POINTER(ScipyCsrF32), POINTER(ScipyCsrF32)] + arg_list[2:],
        )

        arg_list = [
            POINTER(ScipyCsrF32),  # pX
            POINTER(ScipyCscF32),  # pW
            c_uint64,  # len
            POINTER(c_uint32),  # X_row_idx
            POINTER(c_uint32),  # W_col_idx
            POINTER(c_float),  # val
            c_int,  # threads
        ]
        corelib.fillprototype(
            self.clib_float32.c_sparse_inner_products_csr_f32,
            None,
            [POINTER(ScipyCsrF32)] + arg_list[1:],
        )
        corelib.fillprototype(
            self.clib_float32.c_sparse_inner_products_drm_f32,
            None,
            [POINTER(ScipyDrmF32)] + arg_list[1:],
        )

    def sparse_matmul(self, X, Y, eliminate_zeros=False, sorted_indices=True, threads=-1):
        """
        Sparse-Sparse matrix multiplication with multithreading (shared-memory).

        Args:
            X (smat.csc_matrix, smat.csr_matrix, ScipyCscF32, ScipyCsrF32): The first sparse matrix.
            Y (smat.csc_matrix, smat.csr_matrix, ScipyCscF32, ScipyCsrF32): The second sparse matrix.
            eliminate_zeros (bool, optional): if true, then eliminate (potential) zeros created by maxnnz in output matrix Z. Default is false.
            sorted_indices (bool, optional): if true, then sort the Z.indices for the output matrix Z. Default is true.
            threads (int, optional): The number of threads. Default -1 to use all cores.

        Return:
            matmul_res (smat.csc_matrix or smat.csr_matrix): The matrix multiplication results of X and Y
        """

        if X.shape[1] != Y.shape[0]:
            raise ValueError("X.shape[1]={} != Y.shape[0]={}".format(X.shape[1], Y.shape[0]))

        clib = self.clib_float32
        pred_alloc = ScipyCompressedSparseAllocator()

        def is_col_major(X):
            return isinstance(X, smat.csc_matrix) or isinstance(X, ScipyCscF32)

        def is_row_major(X):
            return isinstance(X, smat.csr_matrix) or isinstance(X, ScipyCsrF32)

        if is_col_major(X) and is_col_major(Y):
            pX = ScipyCscF32.init_from(X)
            pY = ScipyCscF32.init_from(Y)
            clib.c_sparse_matmul_csc_f32(
                pX, pY, pred_alloc.cfunc, eliminate_zeros, sorted_indices, threads
            )
        elif is_row_major(X) and is_row_major(Y):
            pX = ScipyCsrF32.init_from(X)
            pY = ScipyCsrF32.init_from(Y)
            clib.c_sparse_matmul_csr_f32(
                pX, pY, pred_alloc.cfunc, eliminate_zeros, sorted_indices, threads
            )
        elif is_col_major(X) and is_row_major(Y):
            if X.nnz > Y.nnz:
                Y = Y.tocsc()
                pX = ScipyCscF32.init_from(X)
                pY = ScipyCscF32.init_from(Y)
                clib.c_sparse_matmul_csc_f32(
                    pX, pY, pred_alloc.cfunc, eliminate_zeros, sorted_indices, threads
                )
            else:
                X = X.tocsr()
                pX = ScipyCsrF32.init_from(X)
                pY = ScipyCsrF32.init_from(Y)
                clib.c_sparse_matmul_csr_f32(
                    pX, pY, pred_alloc.cfunc, eliminate_zeros, sorted_indices, threads
                )
        elif is_row_major(X) and is_col_major(Y):
            if X.nnz > Y.nnz:
                Y = Y.tocsr()
                pX = ScipyCsrF32.init_from(X)
                pY = ScipyCsrF32.init_from(Y)
                clib.c_sparse_matmul_csr_f32(
                    pX, pY, pred_alloc.cfunc, eliminate_zeros, sorted_indices, threads
                )
            else:
                X = X.tocsc()
                pX = ScipyCscF32.init_from(X)
                pY = ScipyCscF32.init_from(Y)
                clib.c_sparse_matmul_csc_f32(
                    pX, pY, pred_alloc.cfunc, eliminate_zeros, sorted_indices, threads
                )
        else:
            raise ValueError(
                "X and Y should be either csr_matrix/csc_matrix/ScipyCscF32/ScipyCsrF32 !"
            )

        return pred_alloc.get()

    def sparse_inner_products(self, pX, pW, X_row_idx, W_col_idx, pred_values=None, threads=-1):
        """
        Sparse-Sparse matrix batch inner product with multithreading (shared-memory).
        Do inner product for rows from `pX` indicated by `X_row_idx`, and columns from `pW` indicated by `W_col_idx`.
        Results will be written in `pred_values` if provided; Otherwise, create a new array for results.

        Args:
            pX (ScipyCsrF32, ScipyDrmF32): The first sparse matrix.
            pW (ScipyCscF32, ScipyDcmF32): The second sparse matrix.
            X_row_idx (ndarray): Row indexes for `pX`.
            W_col_idx (ndarray): Column indexes for `pW`.
            pred_values (ndarray, optional): The inner product result array.
            threads (int, optional): The number of threads. Default -1 to use all cores.

        Return:
            pred_values (ndarray): The matrix batch inner product results.
                If `pred_values` not given, return a new allocated ndarray, dtype same as `pW`.
        """
        clib = self.clib_float32

        nnz = len(X_row_idx)
        assert nnz == len(W_col_idx)

        if not isinstance(pW, ScipyCscF32):
            raise NotImplementedError("type(pW) = {} no implemented".format(type(pW)))

        if isinstance(pX, ScipyCsrF32):
            c_sparse_inner_products = clib.c_sparse_inner_products_csr_f32
        elif isinstance(pX, ScipyDrmF32):
            c_sparse_inner_products = clib.c_sparse_inner_products_drm_f32
        else:
            raise NotImplementedError("type(pX) = {} no implemented".format(type(pX)))

        if pred_values is None or len(pred_values) != nnz or pred_values.dtype != np.float32:
            pred_values = np.zeros(nnz, pW.dtype)

        c_sparse_inner_products(
            byref(pX),
            byref(pW),
            nnz,
            X_row_idx.ctypes.data_as(POINTER(c_uint32)),
            W_col_idx.ctypes.data_as(POINTER(c_uint32)),
            pred_values.ctypes.data_as(POINTER(c_float)),
            threads,
        )
        return pred_values

    def link_clustering(self):
        """
        Specify C-lib's clustering method argument and return type.
        """
        arg_list = [
            POINTER(ScipyCsrF32),
            c_uint32,
            c_uint32,
            c_int,
            c_uint32,
            c_int,
            POINTER(c_uint32),
        ]
        corelib.fillprototype(
            self.clib_float32.c_run_clustering_csr_f32, None, [POINTER(ScipyCsrF32)] + arg_list[1:]
        )
        corelib.fillprototype(
            self.clib_float32.c_run_clustering_drm_f32, None, [POINTER(ScipyDrmF32)] + arg_list[1:]
        )

    def run_clustering(
        self,
        py_feat_mat,
        depth,
        algo,
        seed,
        codes=None,
        kmeans_max_iter=20,
        threads=-1,
    ):
        """
        Run clustering with given label embedding matrix and parameters in C++.

        Args:
            py_feat_mat (ScipyCsrF32, ScipyDrmF32): label embedding matrix. (num_labels x num_features).
            depth (int): Depth of K-means clustering N-nary tree.
            algo (str): The algorithm for clustering, either `KMEANS` or `SKMEANS`.
            seed (int): Randoms seed.
            codes (ndarray, optional): Label clustering results.
            kmeans_max_iter (int, optional): Maximum number of iter for reordering each node based on score.
            threads (int, optional): The number of threads. Default -1 to use all cores.

        Return:
            codes (ndarray): The clustering result.
                If `codes` not given, return a new allocated ndarray, dtype `np.uint32`.
        """
        clib = self.clib_float32
        if isinstance(py_feat_mat, ScipyCsrF32):
            run_clustering = clib.c_run_clustering_csr_f32
        elif isinstance(py_feat_mat, ScipyDrmF32):
            run_clustering = clib.c_run_clustering_drm_f32
        else:
            raise NotImplementedError(
                "type(py_feat_mat) = {} no implemented".format(type(py_feat_mat))
            )

        if codes is None or len(codes) != py_feat_mat.shape[0] or codes.dtype != np.uint32:
            codes = np.zeros(py_feat_mat.rows, dtype=np.uint32)
        run_clustering(
            byref(py_feat_mat),
            depth,
            algo,
            seed,
            kmeans_max_iter,
            threads,
            codes.ctypes.data_as(POINTER(c_uint32)),
        )
        return codes

    def link_tfidf_vectorizer(self):
        """
        Specify C-lib's Tfidf vectorizer method argument and return type.
        """
        res_list = c_void_p
        arg_list = [c_char_p]
        corelib.fillprototype(self.clib_float32.c_tfidf_load, res_list, arg_list)

        arg_list = [c_void_p, c_char_p]
        corelib.fillprototype(self.clib_float32.c_tfidf_save, None, arg_list)

        arg_list = [c_void_p]
        corelib.fillprototype(self.clib_float32.c_tfidf_destruct, None, arg_list)

        arg_list = [
            c_int,  # threads
            ScipyCompressedSparseAllocator.CFUNCTYPE,  # pred_alloc for result
        ]

        # model, fname, fname_len, buffer_size
        corelib.fillprototype(
            self.clib_float32.c_tfidf_predict_from_file,
            None,
            [c_void_p, c_void_p, c_uint64, c_uint64] + arg_list,
        )

        # model, corpus, doc_lens, nr_docs
        corelib.fillprototype(
            self.clib_float32.c_tfidf_predict,
            None,
            [c_void_p, c_void_p, POINTER(c_uint64), c_uint64] + arg_list,
        )

        res_list = c_void_p

        # file-list, fname_lens, nr_files, param, buffer_size, threads
        corelib.fillprototype(
            self.clib_float32.c_tfidf_train_from_file,
            res_list,
            [c_void_p, POINTER(c_uint64), c_uint64, POINTER(TfidfVectorizerParam), c_uint64, c_int],
        )
        # corpus, doc_lens, nr_docs, params, threads
        corelib.fillprototype(
            self.clib_float32.c_tfidf_train,
            res_list,
            [c_void_p, POINTER(c_uint64), c_uint64, POINTER(TfidfVectorizerParam), c_int],
        )

    def tfidf_destruct(self, model):
        """
        Destruct Tfdif model.

        Args:
            model (ptr): Pointer to C Tfdif model.
        """
        if type(model) == c_void_p:
            self.clib_float32.c_tfidf_destruct(model)

    def tfidf_save(self, model, save_dir):
        """
        Save trained tfidf vectorizer to disk.

        Args:
            save_dir (str): Folder to save the model.
        """
        self.clib_float32.c_tfidf_save(model, c_char_p(save_dir.encode("utf-8")))

    def tfidf_load(self, load_dir):
        """
        Load a CppTfidf vectorizer from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            pointer to C instance tfidf::Vectorizer
        """
        return self.clib_float32.c_tfidf_load(c_char_p(load_dir.encode("utf-8")))

    def tfidf_train(self, trn_corpus, config=None):
        """
        Train on a corpus.

        Args:
            trn_corpus (list of str or str): Training corpus in the form of a list of strings or path to corpus file/folder.
            config (dict): Dict with keyword arguments to pass to C++ class tfidf::Vectorizer. None to use default in TfidfVectorizerParam.
                For TfidfVectorizerParam, the config should contain
                    base_vect_configs (List(Dict)): list of config (list[TfidfBaseVectorizerParam]) to be used for TfidfBaseVectorizerParam.
                    norm_p (int): after ensembling feature sub matrices, do row-wise normalization with norm_p.
                    buffer_size (int): if train from file, number of bytes allocated for file I/O. Set to 0 to use default value.
                    threads (int): number of threads to use, set to negative to use all
                For TfidfBaseVectorizerParam, the config should contain
                    ngram_range (tuple of int): (min_ngram, max_ngram)
                    truncate_length (int): sequence truncation length, set to negative to disable
                    max_feature (int): maximum number of features allowed, set to 0 to disable
                    min_df_ratio (float, [0, max_df_ratio)): min ratio for document frequency truncation
                    max_df_ratio (float, (min_df_ratio, 1]): max ratio for document frequency truncation
                    min_df_cnt (int, [0, max_df_cnt)): min count for document frequency truncation
                    max_df_cnt (float, (min_df_cnt, Inf)): max count for document frequency truncation. Default -1 to disable.
                    binary (bool): whether to binarize term frequency, default False
                    use_idf (bool): whether to use inverse document frequency, default True
                    smooth_idf (bool): whether to smooth IDF by adding 1 to all DF counts, default True
                    add_one_idf (bool): whether to smooth IDF by adding 1 to all IDF scores, default False
                    sublinear_tf (bool): whether to use sublinear mapping (log) on term frequency, default False
                    keep_frequent_feature (bool): if max_feature > 0, will only keep max_feature features by
                                    ignoring features with low document frequency (if True, default),
                                    ignoring features with high document frequency (if False)
                    norm (str, 'l1' or 'l2'): feature vector will have unit l1 or l2 norm
                    analyzer (str, 'word', 'char' or 'char_wb'): Whether to use word or character n-grams.
                                    Option ‘char_wb’ creates character n-grams only from text inside word boundaries,
                                    n-grams at the edges of words are padded with space.
                    buffer_size (int): if train from file, number of bytes allocated for file I/O. Set to 0 to use default value.
                    threads (int): number of threads to use, set to negative to use all

        Returns:
            pointer to C instance tfidf::Vectorizer
        """

        # Check whether "base_vect_configs" is in config.keys()
        # If not, this config is for TfidfBaseVectorizerParam.
        # Otherwise, this config is for TfidfVectorizerParam.
        if "base_vect_configs" not in config:
            base_vect_param_list = [TfidfBaseVectorizerParam(config)]
            norm_p = base_vect_param_list[0].norm_p
        else:
            base_vect_param_list = [
                TfidfBaseVectorizerParam(base_vect_config)
                for base_vect_config in config["base_vect_configs"]
            ]
            norm_p = config["norm_p"]
        params = TfidfVectorizerParam(base_vect_param_list, norm_p)

        if isinstance(trn_corpus, str):
            if os.path.isfile(trn_corpus):  # train from a single corpus file
                corpus_files = [trn_corpus]
            elif os.path.isdir(trn_corpus):  # train from a folder of corpus files
                corpus_files = [
                    os.path.join(trn_corpus, f)
                    for f in sorted(os.listdir(trn_corpus))
                    if os.path.isfile(os.path.join(trn_corpus, f))
                ]
            else:
                raise Exception("Failed to load training corpus from {}".format(trn_corpus))
            nr_files = len(corpus_files)
            c_corpusf_arr = (c_char_p * nr_files)()
            c_corpusf_arr[:] = [line.encode("utf-8") for line in corpus_files]
            fname_lens = np.array([len(line) for line in c_corpusf_arr], dtype=np.uint64)

            model = self.clib_float32.c_tfidf_train_from_file(
                c_corpusf_arr,
                fname_lens.ctypes.data_as(POINTER(c_uint64)),
                nr_files,
                params,
                config["buffer_size"],
                config["threads"],
            )
        else:
            nr_doc = len(trn_corpus)
            c_corpus_arr = (c_char_p * nr_doc)()
            c_corpus_arr[:] = [line.encode("utf-8") for line in trn_corpus]
            doc_lens = np.array([len(line) for line in c_corpus_arr], dtype=np.uint64)

            model = self.clib_float32.c_tfidf_train(
                c_corpus_arr,
                doc_lens.ctypes.data_as(POINTER(c_uint64)),
                nr_doc,
                params,
                config["threads"],
            )

        return model

    def tfidf_predict(self, model, corpus, buffer_size=0, threads=-1):
        """
        Vectorize a corpus.

        Args:
            model (ctypes.c_void_p): pointer to tfidf::Vectorizer model
            corpus (list): List of strings to vectorize.
            buffer_size (int, default 0): number of bytes used for file I/O while train from file, set to 0 to use default value
            threads (int, default -1): number of threads to use for predict, set to negative to use all

        Returns:
            scipy.sparse.csr.csr_matrix: Matrix of features.
        """
        pred_alloc = ScipyCompressedSparseAllocator()
        if isinstance(corpus, str):
            # train from file
            assert os.path.isfile(corpus), "Cannot predict from {}!".format(corpus)
            corpus_utf8 = corpus.encode("utf-8")

            self.clib_float32.c_tfidf_predict_from_file(
                model,
                c_char_p(corpus_utf8),
                len(corpus_utf8),
                buffer_size,
                threads,
                pred_alloc.cfunc,
            )

        else:
            # in memory predict
            nr_doc = len(corpus)
            c_corpus_arr = (c_char_p * nr_doc)()
            c_corpus_arr[:] = [line.encode("utf-8") for line in corpus]
            doc_lens = np.array([len(line) for line in c_corpus_arr], dtype=np.uint64)

            self.clib_float32.c_tfidf_predict(
                model,
                c_corpus_arr,
                doc_lens.ctypes.data_as(POINTER(c_uint64)),
                nr_doc,
                threads,
                pred_alloc.cfunc,
            )
        return pred_alloc.get()

    def link_ann_hnsw_methods(self):
        """
        Specify C-lib's ANN HNSW method argument and return type.
        """
        data_type_map = {"drm": POINTER(ScipyDrmF32), "csr": POINTER(ScipyCsrF32)}
        metric_type_list = ["ip", "l2"]
        self.ann_hnsw_fn_dict = {}
        for data_type in data_type_map:
            for metric_type in metric_type_list:
                local_fn_dict = {"data_type": data_type, "metric_type": metric_type}

                fn_name = "train"
                c_fn_name = f"c_ann_hnsw_{fn_name}_{data_type}_{metric_type}_f32"
                local_fn_dict[fn_name] = getattr(self.clib_float32, c_fn_name)
                res_list = c_void_p  # pointer to C/C++ pecos::ann::HNSW
                arg_list = [
                    data_type_map[data_type],
                    c_uint32,  # M
                    c_uint32,  # efC
                    c_int,  # threads
                    c_int,  # max_level_upper_bound
                ]
                corelib.fillprototype(local_fn_dict[fn_name], res_list, arg_list)

                fn_name = "load"
                c_fn_name = f"c_ann_hnsw_{fn_name}_{data_type}_{metric_type}_f32"
                local_fn_dict[fn_name] = getattr(self.clib_float32, c_fn_name)
                res_list = c_void_p  # pointer to C/C++ pecos::ann::HNSW
                arg_list = [c_char_p]  # pointer to char* model_dir
                corelib.fillprototype(local_fn_dict[fn_name], res_list, arg_list)

                fn_name = "save"
                c_fn_name = f"c_ann_hnsw_{fn_name}_{data_type}_{metric_type}_f32"
                local_fn_dict[fn_name] = getattr(self.clib_float32, c_fn_name)
                res_list = None
                arg_list = [
                    c_void_p,  # pointer to C/C++ pecos::ann::HNSW
                    c_char_p,  # pointer to char* model_dir
                ]
                corelib.fillprototype(local_fn_dict[fn_name], res_list, arg_list)

                fn_name = "destruct"
                c_fn_name = f"c_ann_hnsw_{fn_name}_{data_type}_{metric_type}_f32"
                local_fn_dict[fn_name] = getattr(self.clib_float32, c_fn_name)
                res_list = None
                arg_list = [c_void_p]  # pointer to C/C++ pecos::ann::HNSW
                corelib.fillprototype(local_fn_dict[fn_name], res_list, arg_list)

                fn_name = "searchers_create"
                c_fn_name = f"c_ann_hnsw_{fn_name}_{data_type}_{metric_type}_f32"
                local_fn_dict[fn_name] = getattr(self.clib_float32, c_fn_name)
                res_list = c_void_p  # pointer to C/C++ std::vector<pecos::ann::HNSW::Searcher>
                arg_list = [
                    c_void_p,  # pointer C/C++ pecos::ann::HNSW
                    c_uint32,  # number of searcher
                ]
                corelib.fillprototype(local_fn_dict[fn_name], res_list, arg_list)

                fn_name = "searchers_destruct"
                c_fn_name = f"c_ann_hnsw_{fn_name}_{data_type}_{metric_type}_f32"
                local_fn_dict[fn_name] = getattr(self.clib_float32, c_fn_name)
                res_list = None
                arg_list = [c_void_p]  # pointer to C/C++ std::vector<pecos::ann::HNSW::Searcher>
                corelib.fillprototype(local_fn_dict[fn_name], res_list, arg_list)

                fn_name = "predict"
                c_fn_name = f"c_ann_hnsw_{fn_name}_{data_type}_{metric_type}_f32"
                local_fn_dict[fn_name] = getattr(self.clib_float32, c_fn_name)
                res_list = None
                arg_list = [
                    c_void_p,
                    data_type_map[data_type],
                    POINTER(c_uint32),  # uint32_t* ret_idx
                    POINTER(c_float),  # float* ret_val
                    c_uint32,  # efS
                    c_uint32,  # topk
                    c_int,  # threads
                    c_void_p,  # pointer to C/C++ std::vector<pecos::ann::HNSW::Searcher>
                ]
                corelib.fillprototype(local_fn_dict[fn_name], res_list, arg_list)

                self.ann_hnsw_fn_dict[data_type, metric_type] = local_fn_dict

    def ann_hnsw_init(self, data_type, metric_type):
        """Python to C/C++ interface for ANN-HNSW initialization
        Args:
            data_type (str): data type for items/query matrices, can be either drm or csr
            metric_type (str): metric type for computing distance functions, can be either ip or l2
        Returns:
            ann_hnsw_fn_dict (dict): a dictionary that holds clib's C/C++ functions for Python to call
        """

        if (data_type, metric_type) not in self.ann_hnsw_fn_dict:
            raise NotImplementedError(
                "data_type={} and metric_type={} is not implemented".format(data_type, metric_type)
            )
        return self.ann_hnsw_fn_dict[data_type, metric_type]


clib = corelib(os.path.join(os.path.dirname(os.path.abspath(pecos.__file__)), "core"), "libpecos")
