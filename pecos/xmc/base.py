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
import json
import logging
import math
import os
from abc import ABCMeta
from glob import glob
from os import path

import dataclasses as dc
import numpy as np
import pecos
import scipy.sparse as smat
from pecos.core import (
    ScipyCompressedSparseAllocator,
    ScipyCscF32,
    ScipyCsrF32,
    ScipyDrmF32,
    clib,
    XLINEAR_INFERENCE_MODEL_TYPES,
)
from pecos.utils import smat_util
from pecos.utils.cluster_util import ClusterChain, hierarchical_kmeans
from sklearn.preprocessing import normalize

LOGGER = logging.getLogger(__name__)


class IndexerMeta(ABCMeta):
    """Metaclass for keeping track of all `Indexer` subclasses."""

    indexer_dict = {}  # type: ignore

    def __new__(cls, name, bases, attr):
        cls = super().__new__(cls, name, bases, attr)
        if cls.__name__ != "Indexer":
            IndexerMeta.indexer_dict[cls.__name__.lower()] = cls
        return cls


class Indexer(metaclass=IndexerMeta):
    """Enables calling `Indexer` subclasses by string names."""

    indexer_dict = IndexerMeta.indexer_dict

    @classmethod
    def gen(cls, feat_mat, indexer_type="hierarchicalkmeans", **kwargs):
        """Generate a cluster chain.

        Args:
            feat_mat (numpy.ndarray or scipy.sparse.csr.csr_matrix): Matrix of label features.
            indexer_type (str): Lower-cased name of the desired `Indexer` subclass to use.
            **kwargs: Keyword arguments to pass to the specified `Indexer` subclass.

        Returns:
            ClusterChain: The generated cluster chain.
        """

        return cls.indexer_dict[indexer_type].gen(feat_mat, **kwargs)


class HierarchicalKMeans(Indexer):
    """Indexer using Hierarchical K-means.

    See more details in Algorithm 1 of PECOS paper (Yu et al., 2020).

    PECOS: Prediction for Enormous and Correlated Output Spaces
        Hsiang-Fu Yu, Kai Zhong, Inderjit S. Dhillon
        https://arxiv.org/abs/2010.05878
    """

    KMEANS = 0  # KMEANS
    SKMEANS = 5  # Spherical KMEANS

    @dc.dataclass
    class TrainParams(pecos.BaseParams):  # type: ignore
        """Training Parameters of Hierarchical K-means.

        nr_splits (int, optional): The out-degree of each internal node of the tree. Ignored if `imbalanced_ratio != 0` because imbalanced clustering supports only 2-means. Default is `16`.
        min_codes (int): The number of direct child nodes that the top level of the hierarchy should have.
        max_leaf_size (int, optional): The maximum size of each leaf node of the tree. Default is `100`.
        imbalanced_ratio (float, optional): Value between `0.0` and `0.5` (inclusive). Indicates how relaxed the balancedness constraint of 2-means can be. Specifically, if an iteration of 2-means is clustering `L` labels, the size of the output 2 clusters will be within approx `imbalanced_ratio * 2 * L` of each other. Default is `0.0`.
        imbalanced_depth (int, optional): Maximum depth of imbalanced clustering. After depth `imbalanced_depth` is reached, balanced clustering will be used. Default is `100`.
        spherical (bool, optional): True will l2-normalize the centroids of k-means after each iteration. Default is `True`.
        seed (int, optional): Random seed. Default is `0`.
        kmeans_max_iter (int, optional): Maximum number of iterations for each k-means problem. Default is `20`.
        threads (int, optional): Number of threads to use. `-1` denotes all CPUs. Default is `-1`.
        """

        nr_splits: int = 16
        min_codes: int = None  # type: ignore
        max_leaf_size: int = 100
        imbalanced_ratio: float = 0.0
        imbalanced_depth: int = 100
        spherical: bool = True
        seed: int = 0
        kmeans_max_iter: int = 20
        threads: int = -1

    @classmethod
    def gen(
        cls,
        feat_mat,
        train_params=None,
        dtype=np.float32,
        **kwargs,
    ):
        """Generate a cluster chain by using hierarchical k-means.

        Args:
            feat_mat (numpy.ndarray or scipy.sparse.csr.csr_matrix): Matrix of label features.
            train_params (HierarchicalKMeans.TrainParams, optional): training parameters for indexing.
            dtype (type, optional): Data type for matrices. Default is `numpy.float32`.
            **kwargs: Ignored.

        Returns:
            ClusterChain: The generated cluster chain.
        """
        if train_params is None:
            train_params = cls.TrainParams.from_dict(kwargs)
        else:
            train_params = cls.TrainParams.from_dict(train_params)

        if train_params.min_codes is None:
            train_params.min_codes = train_params.nr_splits

        LOGGER.debug(
            f"HierarchicalKMeans train_params: {json.dumps(train_params.to_dict(), indent=True)}"
        )

        # use optimized c++ clustering code if doing balanced clustering
        if train_params.imbalanced_ratio == 0:
            nr_instances = feat_mat.shape[0]
            if train_params.max_leaf_size >= nr_instances:
                # no-need to do clustering
                return ClusterChain.from_partial_chain(
                    smat.csc_matrix(np.ones((nr_instances, 1), dtype=np.float32))
                )

            depth = max(1, int(math.ceil(math.log2(nr_instances / train_params.max_leaf_size))))
            if (2 ** depth) > nr_instances:
                raise ValueError(
                    f"max_leaf_size > 1 is needed for feat_mat.shape[0] == {nr_instances} to avoid empty clusters"
                )

            algo = cls.SKMEANS if train_params.spherical else cls.KMEANS

            assert feat_mat.dtype == np.float32
            if isinstance(feat_mat, (smat.csr_matrix, ScipyCsrF32)):
                py_feat_mat = ScipyCsrF32.init_from(feat_mat)
            elif isinstance(feat_mat, (np.ndarray, ScipyDrmF32)):
                py_feat_mat = ScipyDrmF32.init_from(feat_mat)
            else:
                raise NotImplementedError(
                    "type(feat_mat) = {} is not supported.".format(type(feat_mat))
                )

            codes = np.zeros(py_feat_mat.rows, dtype=np.uint32)
            codes = clib.run_clustering(
                py_feat_mat,
                depth,
                algo,
                train_params.seed,
                codes=codes,
                kmeans_max_iter=train_params.kmeans_max_iter,
                threads=train_params.threads,
            )
            C = cls.convert_codes_to_csc_matrix(codes, depth)
            cluster_chain = ClusterChain.from_partial_chain(
                C, min_codes=train_params.min_codes, nr_splits=train_params.nr_splits
            )
        else:
            cluster_chain = hierarchical_kmeans(
                feat_mat,
                max_leaf_size=train_params.max_leaf_size,
                imbalanced_ratio=train_params.imbalanced_ratio,
                imbalanced_depth=train_params.imbalanced_depth,
                spherical=train_params.spherical,
                seed=train_params.seed,
                kmeans_max_iter=train_params.kmeans_max_iter,
                threads=train_params.threads,
            )
            cluster_chain = ClusterChain(cluster_chain)
        return cluster_chain

    @staticmethod
    def convert_codes_to_csc_matrix(codes, depth):
        """Convert a 1d array of cluster assignments into a binary clustering matrix format.
        Args:
            codes (numpy.ndarray): 1d array of integers. Each index of the array corresponds to a label index, each value of the array is the cluster index.
            depth (int): The depth of the hierarchical tree.

        Returns:
            scipy.sparse.csc.csc_matrix: A binary matrix of shape `(len(codes), 1 << depth)`. An entry `(r, c)` in the matrix has value `1` if and only if `codes[r] == c`.
        """

        nr_codes = 1 << depth
        nr_elements = len(codes)

        indptr = np.cumsum(np.bincount(codes + 1, minlength=(nr_codes + 1)), dtype=np.uint64)
        indices = np.argsort(codes * np.float64(nr_elements) + np.arange(nr_elements))
        C = smat_util.csc_matrix(
            (np.ones_like(indices, dtype=np.float32), indices, indptr),
            shape=(nr_elements, nr_codes),
        )
        return C


class Transform(object):
    """The logit transform methods for different post-processors"""

    @staticmethod
    def identity(v, inplace=False):
        """Identical transformation

        Args:
            v (ndarray): The input array
            inplace (bool, optional): Whether to overwrite input for output

        Returns:
            out (ndarray): the transformed result.
        """
        return v

    @staticmethod
    def log_lp_hinge(p, v, inplace=False):
        """Log Lp Hinge transformation

        .. math:: - \\max (1 - v, 0)^p

        Args:
            p (int): The p for Lp formula
            v (ndarray): The input array
            inplace (bool, optional): Whether to overwrite input for output

        Returns:
            out (ndarray): the transformed result.
        """
        if inplace:
            out = v
        else:
            out = np.zeros_like(v)
        out[:] = -(np.maximum(1.0 - v, 0) ** p)
        return out

    @staticmethod
    def lp_hinge(p, v, inplace=False):
        """Lp Hinge transformation

        .. math:: \\exp { - \\max (1 - v, 0)^p }

        Args:
            p (int): The p for Lp formula
            v (ndarray): The input array
            inplace (bool, optional): Whether to overwrite input for output

        Returns:
            out (ndarray): the transformed result.
        """
        out = Transform.log_lp_hinge(p, v, inplace)
        np.exp(out, out=out)
        return out

    @staticmethod
    def get_log_lp_hinge(p):
        """Wrapped function of `log_lp_hinge` for fixed `p`

        Args:
            p (int): The p for Lp formula

        Returns:
            f (function): the `log_lp_hinge` function with fixed `p`.
        """

        def f(v, inplace=False):
            return Transform.log_lp_hinge(p, v, inplace)

        return f

    @staticmethod
    def get_lp_hinge(p):
        """Wrapped function of `lp_hinge` for fixed `p`

        Args:
            p (int): The p for Lp formula

        Returns:
            f (function): the `lp_hinge` function with fixed `p`.
        """

        def f(v, inplace=False):
            return Transform.lp_hinge(p, v, inplace)

        return f

    @staticmethod
    def sigmoid(v, inplace=False):
        """Sigmoid transformation

        .. math:: \\frac{ 1 }{ 1 + \\exp {-v} }

        Args:
            v (ndarray): The input array
            inplace (bool, optional): Whether to overwrite input for output

        Returns:
            out (ndarray): the transformed result.
        """
        if inplace:
            out = v
        else:
            out = np.zeros_like(v)
        out[:] = 1.0 / (1.0 + np.exp(-v))
        return out

    @staticmethod
    def log_sigmoid(v, inplace=False):
        """Log Sigmoid transformation

        .. math:: \\log { \\frac{ 1 }{ 1 + \\exp {-v} } }

        Args:
            v (ndarray): The input array
            inplace (bool, optional): Whether to overwrite input for output

        Returns:
            out (ndarray): the transformed result.
        """
        out = Transform.sigmoid(v, inplace)
        out[:] = np.log(out)
        return out


class Combiner(object):
    """The combining methods for different post-processors"""

    @staticmethod
    def noop(x, y):
        """No operation

        Args:
            x (ndarray): The first input array
            y (ndarray): The second input array

        Returns:
            x (ndarray): The combining method result in `x`.
        """
        return x

    @staticmethod
    def plus(x, y):
        """Plus y to x, write result in x

        Args:
            x (ndarray): The first input array
            y (ndarray): The second input array

        Returns:
            x (ndarray): The combining method result in `x`.
        """
        x[:] += y[:]
        return x

    @staticmethod
    def multiplies(x, y):
        """Multiply x by y, write result in x

        Args:
            x (ndarray): The first input array
            y (ndarray): The second input array

        Returns:
            x (ndarray): The combining method result in `x`.
        """
        x[:] *= y[:]
        return x

    @staticmethod
    def max(x, y):
        """Element-wise maximum of x and y, write result in x

        Args:
            x (ndarray): The first input array
            y (ndarray): The second input array

        Returns:
            x (ndarray): The combining method result in `x`.
        """
        x[:] = np.maximum(x[:], y[:])
        return x


class PostProcessor(object):
    """PostProcessor for chaining the values in the beam search."""

    mapping = {}  # type: ignore

    def __init__(self, transform, combiner):
        """Initialization

        Args:
            transform (function): Transform method.
            combiner (function): Combination method.
        """
        self.transform = transform
        self.combiner = combiner

    @classmethod
    def get(cls, name):
        """Get the PostProcessor instance from name.

        Args:
            name (PostProcessor or str): the post-processor type.

        Returns:
            PostProcessor
        """
        cls.initialize()
        if isinstance(name, cls):
            return name
        else:
            return cls.mapping[name]

    @classmethod
    def initialize(cls):
        """Initialize the mapping between Transform and Combiner."""
        if len(cls.mapping) == 0:
            cls.mapping["noop"] = cls(Transform.identity, Combiner.noop)
            cls.mapping["sigmoid"] = cls(Transform.sigmoid, Combiner.multiplies)
            cls.mapping["log-sigmoid"] = cls(Transform.log_sigmoid, Combiner.plus)

            for p in range(1, 5):
                cls.mapping["l{}-hinge".format(p)] = cls(
                    Transform.get_lp_hinge(p), Combiner.multiplies
                )
                cls.mapping["log-l{}-hinge".format(p)] = cls(
                    Transform.get_log_lp_hinge(p), Combiner.plus
                )

            # for backward compatibility
            cls.mapping[True] = cls.mapping["l3-hinge"]
            cls.mapping[False] = cls.mapping["noop"]
            cls.mapping[None] = cls.mapping["noop"]

    @classmethod
    def valid_list(cls):
        """Get the valid post-processor list

        Returns:
            list of str: valid post-processor types
        """
        cls.initialize()
        return [x for x in cls.mapping.keys() if isinstance(x, str)]


class MLProblem(object):
    """Object containing the X, Y, C, M and R matrices that defines a Multi-Label(ML) problem.

    Creates M from Y*C if not given with multi-threading sparse_matmul.
    Y: shape of N by L, the instance-to-label matrix with binary classification signals
    C: shape of L by K, the label-to-cluster matrix for selecting inst/labels within same cluster
    M: shape of N by K, the instance-to-cluster matrix for negative sampling
    R: shape of N by L, the relevance matrix for cost-sensitive learning

    See more details in Section 3.3.2 of PECOS paper (Yu et al., 2020).

    PECOS: Prediction for Enormous and Correlated Output Spaces
        Hsiang-Fu Yu, Kai Zhong, Inderjit S. Dhillon
        https://arxiv.org/abs/2010.05878
    """

    def __init__(self, X, Y, C=None, M=None, R=None, threads=8):
        """Initialization

        Args:
            X (csr_matrix, np.ndarray or ScipyDrmF32): Instance feature matrix.
            Y (csr_matrix, np.ndarray or ScipyCscF32): Instance-to-label matrix.
            C (csc_matrix, np.ndarray or ScipyCscF32, optional): Label-to-cluster matrix.
                If not given, create an all-one matrix of shape `(Y.shape[1], 1)`.
            M (csc_matrix, np.ndarray or ScipyCscF32, optional): Instance-to-cluster matrix.
                If not given, creates M from Y*C with multi-threading sparse_matmul.
            R (csc_matrix, np.ndarray or ScipyCscF32, optional): Relevance matrix.
                If not given, will use None.
            threads(int, optional): Number of threads for multi-threading. Default to 8.
        """
        dtype = np.float32
        assert X.dtype == dtype
        assert Y.dtype == dtype
        if isinstance(X, (smat.csr_matrix, ScipyCsrF32)):
            self.pX = ScipyCsrF32.init_from(X)
        elif isinstance(X, (np.ndarray, ScipyDrmF32)):
            self.pX = ScipyDrmF32.init_from(X)
        else:
            raise NotImplementedError("type(X) = {} is not supported.".format(type(X)))
        self.pY = ScipyCscF32.init_from(
            Y if isinstance(Y, ScipyCscF32) else Y.tocsc().astype(dtype)
        )

        if R is None or isinstance(R, ScipyCscF32):
            self.pR = ScipyCscF32.init_from(R)
        elif smat.isspmatrix(R):
            self.pR = ScipyCscF32.init_from(R.tocsc().astype(dtype))
        else:
            raise NotImplementedError("type(R) = {} is not supported.".format(type(R)))
        if R is not None:
            # verify R and Y has the same non-zero pattern
            if not np.array_equal(self.pY.buf.indptr, self.pR.buf.indptr):
                raise ValueError("Invalid relevance matrix: Y.indptr != R.indptr")
            if not np.array_equal(self.pY.buf.indices, self.pR.buf.indices):
                raise ValueError("Invalid relevance matrix: Y.indices != R.indices")
            # verify relevance scores are non negative
            if not all(R.data >= 0):
                raise ValueError("Invalid relevance matrix: got value < 0")

        new_C = (
            smat.csc_matrix(np.ones((Y.shape[1], 1), dtype=dtype))
            if C is None
            else C.tocsc().astype(dtype)
        )
        self.pC = ScipyCscF32.init_from(new_C)
        if M is None:
            if C is not None and C.shape[1] > 1:
                # partial chain without M, fall back to Teacher Forcing Negatives: M = Y * C
                new_M = clib.sparse_matmul(Y, C, threads=threads)
                new_M = new_M.tocsc().astype(dtype)
            else:
                new_M = smat.csc_matrix(np.ones((Y.shape[0], 1), dtype=dtype))
            new_M.sort_indices()
            self.pM = ScipyCscF32.init_from(new_M)
        else:
            assert (
                M.shape[0] == Y.shape[0]
            ), f"M.shape[0] = {M.shape[0]} != {Y.shape[0]} = Y.shape[0]"
            assert (
                M.shape[1] == C.shape[1]
            ), f"M.shape[1] = {M.shape[1]} != {C.shape[1]} = C.shape[1]"
            # tocsc() will create additional copy if M is not csc
            # use in-place function sort_indices() to avoid copy
            M = M.tocsc().astype(dtype)
            M.sort_indices()
            self.pM = ScipyCscF32.init_from(M)
        self.dtype = dtype

    @property
    def X(self):
        """The instance feature matrix."""
        return None if self.pX is None else self.pX.buf

    @property
    def Y(self):
        """The label matrix."""
        return None if self.pY is None else self.pY.buf

    @property
    def C(self):
        """The clustering chain matrix."""
        return None if self.pC is None else self.pC.buf

    @property
    def M(self):
        """The matching chain matrix."""
        return None if self.pM is None else self.pM.buf

    @property
    def R(self):
        return None if self.pR is None else self.pR.buf

    @property
    def nr_labels(self):
        """The number of labels"""
        return None if self.pY is None else self.Y.shape[1]


class MLModel(pecos.BaseClass):
    """Linear Multi-Label(ML) model containing weight matrix W and code matrix C.

    See more details in Section 3.3.2 of PECOS paper (Yu et al., 2020).

    PECOS: Prediction for Enormous and Correlated Output Spaces
        Hsiang-Fu Yu, Kai Zhong, Inderjit S. Dhillon
        https://arxiv.org/abs/2010.05878
    """

    @dc.dataclass
    class TrainParams(pecos.BaseParams):
        """Training Parameters of MLModel.

        Attributes:
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
        """

        threshold: float = 0.1
        max_nonzeros_per_label: int = None  # type: ignore
        solver_type: str = "L2R_L2LOSS_SVC_DUAL"
        Cp: float = 1.0
        Cn: float = 1.0
        max_iter: int = 100
        eps: float = 0.1
        bias: float = 1.0
        threads: int = -1
        verbose: int = 0
        newton_eps: float = 0.01

    @dc.dataclass
    class PredParams(pecos.BaseParams):  # type: ignore
        """Prediction Parameters of MLModel

        Attributes:
            only_topk (int, optional): the only topk specified in the model. Default to 20
            post_processor (str, optional):  the post_processor specified in the model. Default to "l3-hinge"
        """

        only_topk: int = 20
        post_processor: str = "l3-hinge"

        def override_with_kwargs(self, pred_kwargs):
            """Override Class attributes from prediction key-word arguments.

            Args:
                pred_kwargs (dict): Args for prediction.

            Returns:
                self (PredParams): Overriden self instance.
            """
            if pred_kwargs is not None:
                if not isinstance(pred_kwargs, dict):
                    raise TypeError("type(pred_kwargs) must be dict")
                overridden_only_topk = pred_kwargs.get("only_topk", None)
                overridden_post_processor = pred_kwargs.get("post_processor", None)
                if overridden_only_topk:
                    self.only_topk = overridden_only_topk
                if overridden_post_processor:
                    self.post_processor = overridden_post_processor
            return self

        def is_valid(self):
            """Check whether self instance is valid"""
            return self.post_processor in PostProcessor.valid_list()

    def __init__(self, W, C=None, bias=-1.0, pred_params=None, **kwargs):
        """Initialization

        Args:
            W(ScipyCscF32 or np.ndarray): Weight matrix.
            C(ScipyCscF32 or np.ndarray, optional): Clustering matrix
            bias(float, optional): The bias of the model. Default to -1.0.
            pred_params(dict): Parameters to initialize `PredParams`.
            **kwargs: Other arguments to override `PredParams`.
        """
        if C is not None:
            if isinstance(C, ScipyCscF32):
                assert C.buf.shape[0] == W.shape[1], "C:{} W:{}".format(C.buf.shape, W.shape)
            else:
                assert C.shape[0] == W.shape[1], "C:{} W:{}".format(C.shape, W.shape)
        else:
            C = smat.csc_matrix(np.ones((W.shape[1], 1), dtype=W.dtype))
        self.pC = ScipyCscF32.init_from(C)
        self.pW = ScipyCscF32.init_from(W)
        self.bias = bias
        pred_params = self.PredParams.from_dict(pred_params)
        pred_params.override_with_kwargs(kwargs.get("pred_kwargs", None))
        self.pred_params = pred_params

    @property
    def C(self):
        """The clustering matrix"""
        return self.pC.buf

    @property
    def W(self):
        """The weight matrix"""
        return None if self.pW is None else self.pW.buf

    @property
    def nr_labels(self):
        """The number of labels"""
        return self.W.shape[1]

    @property
    def nr_codes(self):
        """The number of clusters."""
        return self.C.shape[1]

    @property
    def nr_features(self):
        """The feature dimension"""
        return self.W.shape[0] - (1 if self.bias > 0 else 0)

    @property
    def dtype(self):
        """The model dtype"""
        return self.pW.dtype

    @classmethod
    def load(cls, folder):
        """Load MLModel from file

        Args:
            folder (str): dir from which the model is loaded.

        Returns:
            MLModel
        """
        param = json.loads(open("{}/param.json".format(folder), "r").read())
        assert param["model"] == cls.__name__
        W = smat_util.load_matrix("{}/W.npz".format(folder)).tocsc().sorted_indices()
        C = smat_util.load_matrix("{}/C.npz".format(folder)).tocsc().sorted_indices()
        pred_params = cls.PredParams.from_dict(param["pred_kwargs"])
        return cls(W, C, param["bias"], pred_params)

    @classmethod
    def load_pred_params(cls, folder):
        """Load prediction parameter from file.

        Args:
            folder (str): dir from which the prediction parameter is loaded.

        Returns:
            PredParams
        """
        with open("{}/param.json".format(folder), "r") as fin:
            param = json.loads(fin.read())
        return cls.PredParams.from_dict(param["pred_kwargs"])

    def save(self, folder):
        """Save MLModel to file

        Args:
            folder (str): dir to which the model is saved

        Returns:
            MLModel
        """
        if not path.exists(folder):
            os.makedirs(folder)
        param = {
            "model": self.__class__.__name__,
            "nr_labels": self.nr_labels,
            "nr_features": self.nr_features,
            "nr_codes": self.nr_codes,
            "bias": self.bias,
            "pred_kwargs": self.pred_params.to_dict(),
        }
        param = self.append_meta(param)
        with open("{}/param.json".format(folder), "w") as f:
            f.write(json.dumps(param, indent=True))
        smat_util.save_matrix("{}/W.npz".format(folder), self.W)
        smat_util.save_matrix("{}/C.npz".format(folder), self.C)

    @classmethod
    def train(cls, prob, train_params=None, pred_params=None, **kwargs):
        """Training method for MLModel

        Args:
            prob (MLProblem): the problem to solve
            train_params (TrainParams, optional): instance of TrainParams
            pred_params (PredParams, optional): instance of PredParams
            **kwargs: for backward compatibility of old training interface
                pred_kwargs (dict, optional): prediction kwargs {"only_topk": INT, "post_processor": STR}.
                    If provided, will override pred_params value. Default None to use pred_params's default
        Returns:
            MLModel: the trained MLModel
        """
        if train_params is None:  # for backward compatibility
            train_params = kwargs
        train_params = cls.TrainParams.from_dict(train_params)

        pred_params = cls.PredParams.from_dict(pred_params)
        pred_params.override_with_kwargs(kwargs.get("pred_kwargs", None))
        if not pred_params.is_valid():
            raise ValueError("pred_params is not valid!")

        # Assuming using newton_eps in train_params
        if train_params.solver_type == "L2R_L2LOSS_SVC_PRIMAL":
            train_params.eps = train_params.newton_eps

        model = clib.xlinear_single_layer_train(
            prob.pX,
            prob.pY,
            prob.pC,
            prob.pM,
            prob.pR,
            **train_params.to_dict(),
        )
        return cls(model, prob.pC, train_params.bias, pred_params)

    def get_pred_params(self):
        """Return a deep copy of prediction parameters

        Returns:
            copied_pred_params (dict): Prediction parameters.
        """
        return copy.deepcopy(self.pred_params)

    def predict(
        self,
        X,
        csr_codes=None,
        pred_params=None,
        **kwargs,
    ):
        """Predict on given input data

        Args:
            X (csr_matrix or ndarray): instance feature matrix to predict on
            csr_codes (csr_matrix, optional): the prediction from previous matchers (nr_inst, nr_codes).
                Default None to ignore
            pred_params (MLModel.PredParams, optional): instance of MLModel.PredParams.
                Default None to use the pred_params used in model training.
            kwargs: overriding prediction parameters for backward compatibility
                only_topk (int, optional): override the only topk specified in pred_params
                    Default None to disable overriding
                post_processor (str, optional):  override the post_processor in pred_params
                    Default None to disable overriding
                threads (int, optional): override the number of threads to use for training in pred_params
                    Default to -1 to disable overriding

        Returns:
            pred_csr (csr_matrix): prediction matrix (nr_inst, nr_labels)
        """
        if X.shape[1] != self.nr_features:
            raise ValueError("Feature dimension of query matrix does not match weight matrix")

        pred_params = self.get_pred_params() if pred_params is None else pred_params
        pred_params.override_with_kwargs(kwargs)
        if not pred_params.is_valid():
            raise ValueError("pred_params is not valid!")

        pred_alloc = ScipyCompressedSparseAllocator()

        clib.xlinear_single_layer_predict(
            X,
            csr_codes,
            self.W,
            self.C,
            pred_params.post_processor,
            pred_params.only_topk if pred_params.only_topk else 0,
            kwargs.get("threads", -1),
            self.bias,
            pred_alloc,
        )

        return pred_alloc.get()

    def predict_on_selected_outputs(
        self,
        X,
        selected_outputs_csr,
        csr_codes=None,
        pred_params=None,
        **kwargs,
    ):
        """Predict on given input data

        Args:
            X (csr_matrix or ndarray): instance feature matrix to predict on
            selected_outputs_csr (csr_matrix): the selected outputs to predict
            csr_codes (csr_matrix): the prediction from previous matchers (nr_inst, nr_codes).
                Default None to ignore
            pred_params (MLModel.PredParams, optional): instance of MLModel.PredParams.
                Default None to use the pred_params used in model training.
            kwargs: overriding prediction parameters for backward compatibility
                post_processor (str, optional):  override the post_processor in pred_params
                    Default None to disable overriding
                threads (int, optional): override the number of threads to use for training in pred_params
                    Default to -1 to disable overriding

        Returns:
            pred_csr (csr_matrix): prediction matrix (nr_inst, nr_labels)
        """
        if X.shape[1] != self.nr_features:
            raise ValueError("Feature dimension of query matrix does not match weight matrix")
        if X.shape[0] != selected_outputs_csr.shape[0]:
            raise ValueError("Instance dimension of query and selected output matrix do not match")

        if selected_outputs_csr.shape[1] != self.nr_labels:
            raise ValueError("Label dimension of selected output matrix does not match")

        pred_params = self.get_pred_params() if pred_params is None else pred_params
        pred_params.override_with_kwargs(kwargs)
        if not pred_params.is_valid():
            raise ValueError("pred_params is not valid!")

        pred_alloc = ScipyCompressedSparseAllocator()

        clib.xlinear_single_layer_predict_on_selected_outputs(
            X,
            selected_outputs_csr,
            csr_codes,
            self.W,
            self.C,
            pred_params.post_processor,
            kwargs.get("threads", -1),
            self.bias,
            pred_alloc,
        )

        return pred_alloc.get()

    def get_submodel(self, selected_codes=None, selected_labels=None, reindex=False):
        """Slice/sparsify the model based on connections to given code and labels.

        The purpose of this function is to slice the W and C matrices of a
        given MLModel, such that only the rows in C with connections to
        selected_codes are retained. The rows are further filtered to lie
        within selected_labels.

        Args:
            selected_codes (list of int, optional): labels with non-zeros in these columns will be retained.
                Default None to use all codes
            selected_labels (list of int, optional): labels will be further filtered to lying in this list.
                Default None to use all labels
            reindex (bool, optional):
                if True: shapes of C and W matrices are changed
                if False (default): the shapes do not change but the matrices is sparsified

        Returns:
            dict:{
                'model': MLModel object with sliced/sparsified C and W matrices
                'active_codes': a (sorted) list of codes that are retained
                'active_labels': a (sorted) list of labels that are retained
                }
        """
        if selected_codes is None:
            selected_codes = np.arange(self.nr_codes)
        else:
            if len(selected_codes) > self.nr_codes:
                raise ValueError("Number of codes are more than number of columns in C")
            if max(selected_codes) >= self.nr_codes:
                raise ValueError(
                    "selected_codes out of range for C of shape: {}".format(self.C.shape)
                )
        if selected_labels is None:
            selected_labels = np.arange(self.nr_labels)
        else:
            if len(selected_labels) > self.nr_labels:
                raise ValueError("Number of labels are more than number of rows in C")
            if max(selected_labels) >= self.nr_labels:
                raise ValueError(
                    "selected_labels out of range for C of shape: {}".format(self.C.shape)
                )

        coo = smat_util.get_sparsified_coo(smat.coo_matrix(self.C), selected_labels, selected_codes)
        active_labels = np.sort(np.unique(coo.row))
        active_codes = np.sort(np.unique(coo.col))
        if reindex:
            row_mapping = np.arange(coo.shape[0])
            row_mapping[active_labels] = np.arange(len(active_labels))
            col_mapping = np.arange(coo.shape[1])
            col_mapping[active_codes] = np.arange(len(active_codes))
            new_C = smat.csc_matrix(
                (coo.data, (row_mapping[coo.row], col_mapping[coo.col])),
                shape=(len(active_labels), len(active_codes)),
            )
            new_W = self.W[:, active_labels]
        else:
            new_C = coo.tocsc()
            new_W = smat_util.get_sparsified_coo(
                smat.coo_matrix(self.W),
                np.arange(self.W.shape[0]),
                active_labels,
            ).tocsc()
        return {
            "model": MLModel(C=new_C, W=new_W),
            "active_labels": active_labels,
            "active_codes": active_codes,
        }


class HierarchicalMLModel(pecos.BaseClass):
    """Hierarchical Linear ML Model containing a chain of MLModel.

    See more details in Algorithm 2 of PECOS paper (Yu et al., 2020).

    PECOS: Prediction for Enormous and Correlated Output Spaces
        Hsiang-Fu Yu, Kai Zhong, Inderjit S. Dhillon
        https://arxiv.org/abs/2010.05878
    """

    @dc.dataclass
    class TrainParams(pecos.BaseParams):
        """Training Parameters of HierarchicalMLModel

        Attributes:
            neg_mining_chain (str, optional): negative_sampling_scheme. Also support List/Tuple type for sublayers.
                Default None.
            model_chain (MLModel.TrainParams, optional): MLModel.TrainParams. Also support List/Tuple type of sublayers.
                Default None.
        """

        neg_mining_chain: str = "tfn"
        model_chain: MLModel.TrainParams = None  # type: ignore

    @dc.dataclass
    class PredParams(pecos.BaseParams):
        """Prediction Parameters of HierarchicalMLModel

        Attributes:
            model_chain (MLModel.PredParams, optional): MLModel.PredParams. Also support List/Tuple type for sublayers.
                Default None.
        """

        model_chain: MLModel.PredParams = None  # type: ignore

        def __len__(self):
            return len(self.model_chain)

        def __add__(self, other):
            if not isinstance(other, HierarchicalMLModel.PredParams):
                other = HierarchicalMLModel.PredParams(other)
            ret_model_chain = self.model_chain + other.model_chain
            return HierarchicalMLModel.PredParams(model_chain=ret_model_chain)

        def __getitem__(self, key):
            return HierarchicalMLModel.PredParams(model_chain=self.model_chain[key])

        def override_with_kwargs(self, pred_kwargs):
            """Override Class attributes from prediction key-word arguments.

            Args:
                pred_kwargs (dict): Args for prediction.

            Returns:
                self (PredParams): Overriden self instance.
            """
            if pred_kwargs is None:
                return self
            if not isinstance(pred_kwargs, dict):
                raise TypeError("type(pred_kwargs) must be dict")

            overridden_beam_size = pred_kwargs.get("beam_size", None)
            overridden_only_topk = pred_kwargs.get("only_topk", None)
            overridden_post_processor = pred_kwargs.get("post_processor", None)

            if isinstance(self.model_chain, (list, tuple)):
                depth = len(self.model_chain)
                for d in range(depth):
                    if overridden_beam_size and d < (depth - 1):
                        self.model_chain[d].only_topk = overridden_beam_size
                    if overridden_only_topk and d == (depth - 1):
                        self.model_chain[d].only_topk = overridden_only_topk
                    if overridden_post_processor:
                        self.model_chain[d].post_processor = overridden_post_processor
            elif isinstance(self.model_chain, MLModel.PredParams):
                if overridden_only_topk:
                    self.model_chain.only_topk = overridden_only_topk
                if overridden_post_processor:
                    self.model_chain.post_processor = overridden_post_processor

            return self

    @staticmethod
    def _duplicate_fields_with_name_ending_with_chain(params, cls, depth):
        """Copy cls attributes into destination params with given model chain depth.

        Args:
            params (cls or pecos.BaseParams): Destination params.
            cls (cls): Current cls.
            depth (int): The depth of model chain to copy.

        Returns:
            params (cls or pecos.BaseParams): Copied params.
        """
        if not isinstance(params, cls) or not isinstance(params, pecos.BaseParams):
            raise ValueError("invalid type(params)!")

        for f in dc.fields(cls):
            old_val = getattr(params, f.name)
            if isinstance(old_val, f.type) and f.name.endswith("_chain"):
                setattr(params, f.name, [copy.deepcopy(old_val) for _ in range(depth)])
            elif isinstance(old_val, (list, tuple)):
                if len(old_val) != depth:
                    raise ValueError(f"len(params.{f.name})={len(old_val)} != {depth}")
                if any(not isinstance(cur_param, f.type) for cur_param in old_val):
                    raise ValueError("invalid params!")
        return params

    def __init__(self, model_chain, pred_params=None, is_predict_only=False, **kwargs):
        """Initialization

        Args:
            model_chain (ptr, list or tuple): The model chain.
            pred_params (dict, optional): The prediction parameters. Default to None.
            is_predict_only (bool, optional): Whether to do prediction only and other operations not allowed. Default to False.
            **kwargs: Additional arguments to override `PredParams`
        """
        if isinstance(model_chain, int):
            # model_chain is a C++ object
            # it can only be used for prediction.
            self.model_chain = model_chain
        else:
            if isinstance(model_chain, (list, tuple)):
                self.model_chain = model_chain
            else:
                self.model_chain = [model_chain]

        if pred_params is None:
            pred_params = self.PredParams(
                model_chain=[MLModel.PredParams() for _ in range(len(self.model_chain))],
            )
        else:
            pred_params = pred_params.from_dict(pred_params)
        pred_params.override_with_kwargs(kwargs.get("pred_kwargs", None))
        self.pred_params = pred_params
        self.is_predict_only = is_predict_only

    def __del__(self):
        if self.is_predict_only:
            clib.xlinear_destruct_model(self.model_chain)

    @property
    def depth(self):
        """The model depth"""
        if self.is_predict_only:
            return clib.xlinear_get_int_attr(self.model_chain, "depth")
        else:
            return len(self.model_chain)

    @property
    def nr_features(self):
        """The feature dimension size"""
        if self.is_predict_only:
            return clib.xlinear_get_int_attr(self.model_chain, "nr_features")
        else:
            return self.model_chain[0].nr_features

    @property
    def nr_codes(self):
        """The number of clusters at bottom layer"""
        if self.is_predict_only:
            return clib.xlinear_get_int_attr(self.model_chain, "nr_codes")
        else:
            return self.model_chain[-1].nr_codes

    @property
    def nr_labels(self):
        """The number of labels"""
        if self.is_predict_only:
            return clib.xlinear_get_int_attr(self.model_chain, "nr_labels")
        else:
            return self.model_chain[-1].nr_labels

    def get_weight_matrix_type(self, layer_depth):
        """Get the weight matrix type of a layer at a certain depth

        Args:
            layer_depth (int): The depth of the layer type to get

        Returns:
            weight_matrix_type (string): If is_predict_only=True, the weight matrix type of
                the specified layer is returned. Otherwise if is_predict_only=False, the default
                type "CSC" is returned.

        """
        if self.is_predict_only:
            weight_matrix_value = clib.xlinear_get_layer_type(self.model_chain, layer_depth)
            return list(XLINEAR_INFERENCE_MODEL_TYPES.keys())[
                list(XLINEAR_INFERENCE_MODEL_TYPES.values()).index(weight_matrix_value)
            ]
        else:
            return "CSC"

    def __add__(self, other):
        if self.is_predict_only:
            raise Exception("Model is predict only! __add__ not supported!")
        if not isinstance(other, HierarchicalMLModel):
            other = HierarchicalMLModel(other)
        assert self.model_chain[-1].nr_labels == other.model_chain[0].nr_codes
        ret_model_chain = self.model_chain + other.model_chain
        ret_pred_params = self.pred_params + other.pred_params
        if len(ret_model_chain) != len(ret_pred_params):
            raise ValueError("len(model_chain) != len(pred_params)")
        return HierarchicalMLModel(ret_model_chain, ret_pred_params)

    def __getitem__(self, key):
        if self.is_predict_only:
            raise Exception("Model is predict only! __getitem__ not supported!")
        return HierarchicalMLModel(self.model_chain[key], self.pred_params[key])

    def astype(self, dtype):
        """Cast the models to a specified type

        Args:
            dtype (str or dtype): Typecode or data-type to which the models are cast.

        Returns:
            HierarchicalMLModel
        """
        if self.is_predict_only:
            raise Exception("Model is predict only! astype not supported!")
        if dtype == self.model_chain[0].dtype:
            return self
        else:
            return HierarchicalMLModel(
                [m.astype(dtype) for m in self.model_chain], self.pred_params
            )

    @classmethod
    def load(cls, model_folder, is_predict_only=False, **kwargs):
        """
        Load HierarchicalMLModel from file

        Args:
            model_folder (str): The path to the folder that stores the model.
            is_predict_only (bool): If true, the XLinear model will be loaded in C++ and can only be used for prediction.
            If False, the model is loaded in Python and can be changed such as pruning. The predict-only mode typically
            gives better performance but also means that any method not tied to prediction (e.g., train) cannot be invoked.

            kwargs:
                weight_matrix_type (string): A string determining which type of weight matrix to use.
                    The allowed types are:
                        * "BINARY_SEARCH_CHUNKED": The default, stores the weight matrix in chunked format and
                    uses binary search for vector x chunk products.
                        * "HASH_CHUNKED": Stores the weight matrix in chunked format, but uses a hash table
                    for random row access in the chunks. This hash table is used to compute vector
                    x chunk products.
                        * "CSC": Typically the slowest option. Stories the weight matrix in csc format.
                    This format tends to be the fastest to load.

        Returns:
            HierarchicalMLModel
        """
        param = json.loads(open(f"{model_folder}/param.json", "r", encoding="utf-8").read())
        assert param["model"] == cls.__name__
        depth = int(param.get("depth", len(glob("{}/*.model".format(model_folder)))))

        if is_predict_only:
            model = clib.xlinear_load_predict_only(model_folder, **kwargs)
        else:
            model = [MLModel.load(f"{model_folder}/{d}.model") for d in range(depth)]

        pred_params = cls.PredParams(
            model_chain=[
                MLModel.load_pred_params(f"{model_folder}/{d}.model") for d in range(depth)
            ],
        )
        return cls(model, pred_params=pred_params, is_predict_only=is_predict_only)

    def save(self, folder):
        """Save HierarchicalMLModel to file

        Args:
            folder (str): dir to which the model is saved

        Returns:
            HierarchicalMLModel
        """
        if self.is_predict_only:
            raise Exception("Model is predict only! save not supported!")
        if not path.exists(folder):
            os.makedirs(folder)
        param = {
            "model": self.__class__.__name__,
            "depth": self.depth,
            "nr_features": self.nr_features,
            "nr_codes": self.nr_codes,
            "nr_labels": self.nr_labels,
        }
        param = self.append_meta(param)
        open(f"{folder}/param.json", "w", encoding="utf-8").write(json.dumps(param, indent=True))
        for d in range(self.depth):
            local_folder = f"{folder}/{d}.model"
            self.model_chain[d].save(local_folder)

    @classmethod
    def train(
        cls,
        prob,
        clustering=None,
        relevance_chain=None,
        matching_chain=None,
        train_params=None,
        pred_params=None,
        **kwargs,
    ):
        """Training method for HierarchicalMLModel

        Args:
            prob (MLProblem): the problem to solve
            clustering (ClusterChain or None, optional): cluster chain for the model hierarchy
                Default None for the One-Versus-All problem.
            relevance_chain (list of spmatrix): the relevance_chain for cost sensitive learning.
                skip cost-sensitive learning for level i if relevance_chain[i] is None,
                Default None to ignore.
            matching_chain (list of csr_matrix): the matching_chain generated by user-supplied-negatives.
                Their indices will be added to the negative samples if 'usn' in negative_sampling_scheme.
                Default None to ignore.
            train_params (HierarchicalMLModel.TrainParams, optional): training kwargs for each layer
            pred_params (HierarchicalMLModel.PredParams, optional): prediction kwargs for each layer
            kwargs: containing keyword arguments for the solver. See MLModel.TrainParams
                pred_kwargs (dict, optional): prediction kwargs {"beam_size": INT, "only_topk": INT, "post_processor": STR},
                    Default None to use HierarchicalMLModel.DEFAULT_PRED_KWARGS

        Returns:
            HierarchicalMLModel: the trained HierarchicalMLModel
        """

        if clustering is None or clustering is False:
            depth = 1
            if train_params is None:
                train_params = cls.TrainParams(
                    model_chain=tuple([MLModel.TrainParams.from_dict(kwargs)])
                )
            else:
                train_params = cls.TrainParams.from_dict(train_params)
                train_params = cls._duplicate_fields_with_name_ending_with_chain(
                    train_params, cls.TrainParams, depth
                )

            if pred_params is None:
                pred_params = cls.PredParams(model_chain=tuple([MLModel.PredParams()]))
            else:
                pred_params = cls.PredParams.from_dict(pred_params)
                pred_params = cls._duplicate_fields_with_name_ending_with_chain(
                    pred_params, cls.PredParams, depth
                )
            pred_params.override_with_kwargs(kwargs.get("pred_kwargs", None))

            ml_model = MLModel.train(
                prob,
                train_params=train_params.model_chain[0],
                pred_params=pred_params.model_chain[0],
            )
            return HierarchicalMLModel([ml_model], pred_params=pred_params, is_predict_only=False)

        # assert cluster chain in clustering is valid
        clustering = ClusterChain(clustering)
        assert clustering[-1].shape[0] == prob.nr_labels
        depth = len(clustering)

        # construct train_params
        if train_params is None:  # for backward compatibility
            train_params = cls.TrainParams(
                neg_mining_chain=tuple(
                    [kwargs.get("negative_sampling_scheme", "tfn") for _ in range(depth)]
                ),
                model_chain=tuple([MLModel.TrainParams.from_dict(kwargs) for _ in range(depth)]),
            )
        else:
            train_params = cls.TrainParams.from_dict(train_params)
            if train_params.neg_mining_chain is None:
                train_params.neg_mining_chain = kwargs.get("negative_sampling_scheme", "tfn")
            train_params = cls._duplicate_fields_with_name_ending_with_chain(
                train_params, cls.TrainParams, depth
            )
        train_params.neg_mining_chain = [ns.lower() for ns in train_params.neg_mining_chain]

        # construct pred_params
        if pred_params is None:
            pred_params = cls.PredParams(
                model_chain=tuple([MLModel.PredParams() for _ in range(depth)])
            )
        else:
            pred_params = cls.PredParams.from_dict(pred_params)
            pred_params = cls._duplicate_fields_with_name_ending_with_chain(
                pred_params, cls.PredParams, depth
            )
        pred_params.override_with_kwargs(kwargs.get("pred_kwargs", None))

        LOGGER.debug(
            f"HierarchicalMLModel train_params: {json.dumps(train_params.to_dict(), indent=True)}"
        )
        LOGGER.debug(
            f"HierarchicalMLModel pred_params: {json.dumps(pred_params.to_dict(), indent=True)}"
        )
        # construct Y_chain
        # avoid large matmul_threads to prevent overhead in Y.dot(C) and save memory
        matmul_threads = train_params.model_chain[0].threads
        if matmul_threads <= 0:
            matmul_threads = max(os.cpu_count(), matmul_threads)
        matmul_threads = min(32, matmul_threads)
        Y_chain = [prob.Y]
        for C in reversed(clustering[1:]):
            Y_t = clib.sparse_matmul(Y_chain[-1], C, threads=matmul_threads).tocsc()
            Y_chain.append(Y_t)
        Y_chain.reverse()

        if matching_chain is None:
            matching_chain = [None for _ in range(depth)]
        if relevance_chain is None:
            relevance_chain = [None for _ in range(depth)]

        cur_prob, M_pred = prob, None
        model_chain = []
        for t, (Y, C, R, M_usn) in enumerate(
            zip(Y_chain, clustering, relevance_chain, matching_chain)
        ):
            negative_sampling_scheme = train_params.neg_mining_chain[t]
            cur_train_params = train_params.model_chain[t]
            cur_pred_params = pred_params.model_chain[t]
            LOGGER.info(
                f"Training Layer {t} of {len(Y_chain)} Layers in HierarchicalMLModel, neg_mining={negative_sampling_scheme}.."
            )
            if t == 0:
                M = None
                # if got partial chain, enter hierarchical ranker mode
                if C.shape[1] > 1:
                    shape = (cur_prob.Y.shape[0], C.shape[1])
                    M = smat.csc_matrix(shape, dtype=cur_prob.Y.dtype)
                    if "usn" in negative_sampling_scheme:
                        if M_usn is not None:
                            M += smat_util.binarized(M_usn)
                    if "tfn" in negative_sampling_scheme:
                        M_true = clib.sparse_matmul(Y, C, threads=matmul_threads).tocsc()
                        M += smat_util.binarized(M_true)
            else:
                # Preparing negative sampling for M
                shape = (cur_prob.Y.shape[0], C.shape[1])
                M = smat.csc_matrix(shape, dtype=cur_prob.Y.dtype)
                if "usn" in negative_sampling_scheme:
                    if M_usn is not None:
                        M += smat_util.binarized(M_usn)
                if "tfn" in negative_sampling_scheme:
                    M_true = Y_chain[t - 1].tocsc()
                    M += smat_util.binarized(M_true)
                if any("man" in ns for ns in train_params.neg_mining_chain[t:]):
                    M_pred = model_chain[-1].predict(cur_prob.pX, csr_codes=M_pred)
                if "man" in negative_sampling_scheme:
                    M += smat_util.binarized(M_pred)

            cur_prob = MLProblem(cur_prob.pX, Y, R=R, C=C, M=M, threads=matmul_threads)
            cur_model = MLModel.train(
                cur_prob, train_params=cur_train_params, pred_params=cur_pred_params
            )
            model_chain.append(cur_model)
        return cls(model_chain, pred_params=pred_params, is_predict_only=False)

    def get_pred_params(self):
        return copy.deepcopy(self.pred_params)

    def predict(
        self,
        X,
        csr_codes=None,
        pred_params=None,
        **kwargs,
    ):
        """Predict on given input data

        Args:
            X (csr_matrix or ndarray): instance feature matrix to predict on
            csr_codes (csr_matrix, optional): the prediction from pervious matchers (nr_inst, K0).
            pred_params (HierarchicalMLModel.PredParams, optional): instance of HierarchicalMLModel.PredParams.
                Default None to use the pred_params used in model training.
            kwargs: overriding prediction parameters for backward compatibility
                beam_size (int, optional): override the beam_size specified in pred_params (except last layer)
                    Default None to disable overriding
                only_topk (int, optional): override the only_topk specified in pred_params (only last layer)
                    Default None to disable overriding
                post_processor (str, optional):  override the post_processor specified in pred_params (all layers)
                    Default None to disable overriding
                threads (int, optional): the number of threads to use for training.
                    Defaults to -1 to use all
        Returns:
            pred_csr (csr_matrix): prediction matrix (nr_inst, nr_labels)
        """
        assert X.dtype == np.float32
        assert isinstance(X, smat.csr_matrix) or (
            isinstance(X, np.ndarray) and X.flags["C_CONTIGUOUS"]
        )
        assert X.shape[1] == self.nr_features

        # construct pred_params
        if pred_params is None:
            pred_params = self.get_pred_params()
        elif isinstance(pred_params, self.PredParams):
            pred_params = self.PredParams.from_dict(pred_params)
            pred_params = self._duplicate_fields_with_name_ending_with_chain(
                pred_params, self.PredParams, self.depth
            )
        else:
            raise ValueError("unknown type(pred_params)!!")
        pred_params.override_with_kwargs(kwargs)

        if self.is_predict_only:
            if csr_codes is not None:
                raise NotImplementedError(
                    "is_predict_only=True did not support csr_codes being not None"
                )

            old_chain = self.get_pred_params().model_chain
            new_chain = pred_params.model_chain

            # check if post_processor is valid (support by C++) after overriding
            if all(
                old_p.post_processor == new_p.post_processor
                for (old_p, new_p) in zip(old_chain, new_chain)
            ):
                overridden_post_processor = None
            elif all(new_chain[0].post_processor == new_p.post_processor for new_p in new_chain):
                overridden_post_processor = new_chain[0].post_processor
            else:
                raise NotImplementedError(
                    "when is_predict_only=True, post_processor is not supported for overriddng"
                )

            # check if beam_size is valid (support by C++) after overriding
            if all(
                old_p.only_topk == new_p.only_topk
                for (old_p, new_p) in zip(old_chain[:-1], new_chain[:-1])
            ):
                overridden_beam_size = None
            elif all(new_chain[0].only_topk == new_p.only_topk for new_p in new_chain[:-1]):
                overridden_beam_size = new_chain[0].only_topk
            else:
                raise NotImplementedError(
                    "when is_predict_only=True, beam_size is not supported for overriding"
                )

            # Call C++ code
            pred_alloc = ScipyCompressedSparseAllocator()
            clib.xlinear_predict(
                self.model_chain,
                X,
                overridden_beam_size,
                overridden_post_processor,
                new_chain[-1].only_topk,
                kwargs.get("threads", -1),
                pred_alloc,
            )

            return pred_alloc.get()
        else:
            pred_csr = csr_codes
            for d in range(self.depth):
                cur_model = self.model_chain[d]
                pred_csr = cur_model.predict(
                    X,
                    csr_codes=pred_csr,
                    pred_params=pred_params.model_chain[d],
                    threads=kwargs.get("threads", -1),
                )

            return pred_csr

    def predict_on_selected_outputs(
        self,
        X,
        selected_outputs_csr,
        pred_params=None,
        **kwargs,
    ):
        """Predict on given input data
        Args:
            X (csr_matrix or ndarray): instance feature matrix to predict on
            selected_outputs_csr (csr_matrix): the selected outputs to predict
            pred_params (HierarchicalMLModel.PredParams, optional): instance of HierarchicalMLModel.PredParams.
                Default None to use the pred_params used in model training.
            kwargs: overriding prediction parameters for backward compatibility
                post_processor (str, optional):  override the post_processor specified in pred_params (all layers)
                    Default None to disable overriding
                threads (int, optional): the number of threads to use for training.
                    Defaults to -1 to use all
        Returns:
            pred_csr (csr_matrix): prediction matrix (nr_inst, nr_labels)
        """
        if X.dtype != np.float32:
            raise ValueError("X.dtype = {} is not supported".format(X.dtype))
        if not isinstance(X, smat.csr_matrix) and not (
            isinstance(X, np.ndarray) and X.flags["C_CONTIGUOUS"]
        ):
            raise ValueError("type(X) = {} is not supported".format(type(X)))
        if X.shape[1] != self.nr_features:
            raise ValueError("Feature dimension of query matrix does not match weight matrix")

        if not isinstance(selected_outputs_csr, smat.csr_matrix):
            raise ValueError(
                "type(selected_outputs_csr) = {} is not supported".format(
                    type(selected_outputs_csr)
                )
            )
        if selected_outputs_csr.shape[1] != self.nr_labels:
            raise ValueError("Label dimension of selected output matrix does not match")

        if X.shape[0] != selected_outputs_csr.shape[0]:
            raise ValueError("Instance dimension of query and selected output matrix do not match")

        # construct pred_params
        if pred_params is None:
            pred_params = self.get_pred_params()
        elif isinstance(pred_params, self.PredParams):
            pred_params = self.PredParams.from_dict(pred_params)
            pred_params = self._duplicate_fields_with_name_ending_with_chain(
                pred_params, self.PredParams, self.depth
            )
        else:
            raise ValueError("unknown type(pred_params)!!")
        pred_params.override_with_kwargs(kwargs)

        if self.is_predict_only:
            for layer_depth in range(self.depth):
                if self.get_weight_matrix_type(layer_depth) != "CSC":
                    raise NotImplementedError(
                        "is_predict_only=True not supported for weight_matrix_type = {}".format(
                            self.get_weight_matrix_type(layer_depth)
                        )
                    )

            old_chain = self.get_pred_params().model_chain
            new_chain = pred_params.model_chain

            # check if post_processor is valid (support by C++) after overriding
            if all(
                old_p.post_processor == new_p.post_processor
                for (old_p, new_p) in zip(old_chain, new_chain)
            ):
                overridden_post_processor = None
            elif all(new_chain[0].post_processor == new_p.post_processor for new_p in new_chain):
                overridden_post_processor = new_chain[0].post_processor
            else:
                raise NotImplementedError(
                    "when is_predict_only=True, post_processor is not supported for overriddng"
                )

            # Call C++ code
            pred_alloc = ScipyCompressedSparseAllocator()
            clib.xlinear_predict_on_selected_outputs(
                self.model_chain,
                X,
                selected_outputs_csr,
                overridden_post_processor,
                kwargs.get("threads", -1),
                pred_alloc,
            )

            return pred_alloc.get()
        else:
            selected_outputs_csrs = []
            selected_outputs_csrs.insert(0, selected_outputs_csr)
            for d in range(self.depth - 2, -1, -1):
                prev_csr = clib.sparse_matmul(
                    selected_outputs_csrs[0],
                    self.model_chain[d + 1].C,
                    threads=kwargs.get("threads", -1),
                ).tocsr()
                selected_outputs_csrs.insert(0, prev_csr)

            prev_pred_csr = None
            for d in range(self.depth):
                prev_pred_csr = self.model_chain[d].predict_on_selected_outputs(
                    X=X,
                    selected_outputs_csr=selected_outputs_csrs[d],
                    csr_codes=prev_pred_csr,
                    pred_params=pred_params.model_chain[d],
                    threads=kwargs.get("threads", -1),
                )

            return prev_pred_csr

    def set_output_constraint(self, labels_to_keep):
        """
        Prune clustering tree to only output labels in labels_to_keep set.
        C matrices denote connectivities between nodes in a level and nodes in level below.
        C matrices are tall, rows are nodes in level below, columns are nodes in level above. Stored in CSC format.
        Update C matrices from bottom-up.

        Args:
            labels_to_keep (iterable over int): label indices to keep
        """
        if self.is_predict_only:
            raise Exception("Model is predict only! set_output_constraint not supported!")
        try:
            labels_to_keep = set(labels_to_keep)
        except TypeError:
            raise TypeError("can not convert labels_to_keep as set variable type!")

        for cur_dep, cur_model in enumerate(self.model_chain[::-1]):
            if len(labels_to_keep) == cur_model.C.shape[0]:
                # keep all labels / nodes
                # so keep all branches in levels above
                break
            for i, label in enumerate(cur_model.C.indices):
                if label not in labels_to_keep:
                    cur_model.C.data[i] = 0
            cur_model.C.eliminate_zeros()
            labels_to_keep = set(
                np.nonzero((cur_model.C.indptr[1:] - cur_model.C.indptr[:-1]) != 0)[0]
            )

    def get_submodel_rooted_at(self, given_depth, child_node_id, reindex=False):
        """
        Get a HierarchicalMLModel sub-model which represents the subtree of child_node_id and at depth: given_depth.

        Args:
            given_depth (int): depth of the model chain
            child_node_id (int): index of the node whose subtree needs to be fetched
            reindex (bool, optional):
                if True: shapes of C and W matrices are changed
                if False (default): the shapes do not change but the matrices is sparsified

        Returns:
            submodel (HierarchicalMLModel): subtree as HierarchicalMLModel object
            label_mapping (MLModel): mapping of indices of the final model to original labels (or None if reindex = False)
        """
        if self.is_predict_only:
            raise Exception("Model is predict only! get_submodel_rooted_at not supported!")

        subtree_chain = []
        parents = [child_node_id]
        for d in range(given_depth, len(self.model_chain)):
            sub_model = self.model_chain[d].get_submodel(selected_codes=parents, reindex=reindex)
            if d == given_depth and not reindex:
                sub_model["model"] = MLModel(
                    C=sub_model["model"].C[:, parents], W=sub_model["model"].W
                )
            parents = sub_model["active_labels"]
            subtree_chain.append(sub_model["model"])
        submodel = HierarchicalMLModel(subtree_chain, pred_params=self.pred_params)
        label_mapping = parents if reindex else None
        return submodel, label_mapping

    def split_model_at_depth(self, given_depth, reindex=False):
        """Splits model tree at a given depth to yield separate models.

        Args:
            given_depth (int): depth at which the model needs to be split.
            reindex (bool, optional):
                if True: shapes of C and W matrices are changed
                if False (default): the shapes do not change but the matrix is sparsified

        Returns:
            a dictionary containing the following keys:
                "parent_model":
                    a HierarchicalMLModel that has model_chain of the
                    original model till depth: given_depth
                "child_models":
                    A list of child model tuples where each tuple has:
                        HierarchicalMLModel corresponding to the subtree at child id corresponding to list index
                        mapping to original labels from the child model output (None if reindex is False)

        Notes:
            As an example if the model tree is a complete binary tree of depth 2,
            and we call this function with given_depth = 1, this is the output (reindex = True)
            "parent_model": 2*1 HierarchicalMLModel having the top-layer of the original tree
            "child_models"[0]:
                model corresponding to child tree of node 0 at height 1
                list([0, 1])
            "child_models"[1]:
                model corresponding to child tree of node 1 at height 1
                list([2, 3])
        """
        if self.is_predict_only:
            raise Exception("Model is predict only! split_model_at_depth not supported!")

        if given_depth < 1:
            raise ValueError("depth has to be a positive integer")
        if given_depth > len(self.model_chain) - 1:
            raise ValueError("depth has to be one less than length of model_chain")
        parent_model = self[:given_depth]
        child_models = []
        for i in range(self.model_chain[given_depth].nr_codes):
            subtree_model, mapping_to_labels = self.get_submodel_rooted_at(given_depth, i, reindex)
            child_models.append((subtree_model, mapping_to_labels))
        return {"parent_model": parent_model, "child_models": child_models}


class LabelEmbeddingFactory(object):
    @staticmethod
    def create(Y=None, X=None, method="pifa", **kwargs):
        """Create label embedding

        Args:
            Y (smat.spmatrix): label matrix (num_samples, num_labels).
            X (smat.csr_matrix or np.ndarray): input feature matrix (num_samples x num_features)
            method (string): label embedding method. (default pifa)
            kwargs:
                Z (smat.csr_matrix or np.ndarray): label feature matrix (num_samples x num_label_features)
                alpha (float, int or np.ndarray): weight(s) for pifa. The value(s) of alpha should be between 0.0 and 1.0.
                threads (int): number of threads for doing sparse matrix multiplication in parallel.
                normalized_Y (bool): if true, the rows of Y will be l2-normalized.
        Returns:
            label_embedding: embedding matrix. (num_labels x num_features)
        """

        mapping = {
            "pifa": LabelEmbeddingFactory.pifa,
            "pifa_lf_concat": LabelEmbeddingFactory.pifa_lf_concat,
            "pifa_lf_convex_combine": LabelEmbeddingFactory.pifa_lf_convex_combine,
        }
        if method.lower() in mapping:
            return mapping[method.lower()](Y, X, **kwargs)
        else:
            assert NotImplementedError(
                f"Label embedding method '{method}' is not implemented. Valid ones: {mapping.keys()}"
            )

    @staticmethod
    def pifa(Y, X, threads=-1, normalized_Y=True):
        """Create pifa embedding

        Args:
            Y (smat.spmatrix): label matrix (num_samples, num_labels).
            X (smat.csr_matrix or np.ndarray): input feature matrix (num_samples x num_features)
            threads (int): number of threads for doing sparse matrix multiplication in parallel.
            normalized_Y (bool): if true, the rows of Y will be l2-normalized.
        Returns:
            label_embedding: pifa embeddings. (num_labels x num_features)
            The returned format will be the same as X.
        """

        if not isinstance(Y, smat.spmatrix):
            raise NotImplementedError("type(Y) should be scipy.sparse.spmatrix")

        if normalized_Y:
            Y = normalize(Y, axis=1, norm="l2")
        YT = Y.T.tocsr()

        if isinstance(X, smat.csr_matrix):
            # YT is csr, X is csr.
            # Note that ScipyCsrF32.init_from() will copy data (incur additional memory),
            # if input matrix's indptr/indices/data dtype is different from its expected dtype.
            pYT = ScipyCsrF32.init_from(YT)
            pX = ScipyCsrF32.init_from(X)
            label_embedding = clib.sparse_matmul(
                pYT, pX, eliminate_zeros=False, sorted_indices=True, threads=threads
            )
        elif isinstance(X, np.ndarray):
            # YT is csr, X is row-major dense nparray
            # Thus, the scipy matmul will not create extra memory
            label_embedding = YT.dot(X)
            label_embedding = np.ascontiguousarray(label_embedding)
        else:
            raise NotImplementedError("type(X) should be row-major spmatrix or ndarray")

        label_embedding = normalize(label_embedding, axis=1, norm="l2", copy=False)
        return label_embedding

    @staticmethod
    def pifa_lf_concat(Y, X, Z, threads=-1, normalized_Y=True):
        """Create label embedding by concatenating pifa embedding and provided existing label embedding

        Args:
            Y (smat.spmatrix): label matrix (num_samples, num_labels).
            X (smat.csr_matrix or np.ndarray): input feature matrix (num_samples x num_features)
            Z (smat.csr_matrix or np.ndarray): existing label feature matrix (num_labels x num_label_features)
            threads (int): number of threads for doing sparse matrix multiplication in parallel.
            normalized_Y (bool): if true, the rows of Y will be l2-normalized.

        Returns:
            label_embedding: pifa_lf_concat embeddings. (num_labels x (num_features + num_label_features)).
            The returned format will be csr_matrix if either X or Z is csr_matrix. Otherwise, it will be ndarray.
        """

        pifa = LabelEmbeddingFactory.pifa(Y, X, threads=threads, normalized_Y=normalized_Y)
        if not isinstance(pifa, (smat.csr_matrix, np.ndarray)) or not isinstance(
            Z, (smat.csr_matrix, np.ndarray)
        ):
            raise NotImplementedError(
                "type(pifa) and type(Z) should be row-major spmatrix or ndarray"
            )
        if isinstance(pifa, np.ndarray) and isinstance(Z, np.ndarray):
            label_embedding = np.hstack([pifa, Z])
        else:
            if isinstance(pifa, np.ndarray):
                pifa = smat_util.dense_to_csr(pifa)
            if isinstance(Z, np.ndarray):
                Z = smat_util.dense_to_csr(Z)
            label_embedding = smat_util.hstack_csr([pifa, Z])
        return label_embedding

    @staticmethod
    def pifa_lf_convex_combine(Y, X, Z, alpha=0.5, threads=-1, normalized_Y=True):
        """Create label embedding by doing convex-combination of pifa embedding and existing label embedding.

        To use this function, Input feature (X) dimension should be the same as existing label feature (Z) dimension.

        Args:
            Y (smat.spmatrix): label matrix (num_samples, num_labels).
            X (smat.csr_matrix or np.ndarray): input feature matrix (num_samples x num_features)
            Z (smat.csr_matrix or np.ndarray): existing label feature matrix (num_labels x num_features)
            alpha (float, int or np.ndarray): weight(s) for pifa. The value(s) of alpha should be between 0.0 and 1.0.
                The returned label embedding is (alpha * pifa + (1 - alpha) * Z).
                If alpha is ndarray, it should be a 1-d array and the length should be num_labels.
            threads (int): number of threads for doing sparse matrix multiplication in parallel.
            normalized_Y (bool): if true, the rows of Y will be l2-normalized.
        Returns:
            label_embedding: pifa_lf_convex_combine embeddings. (num_labels x num_features).
            The returned format will be ndarray if either X or Z is ndarray. Otherwise, it will be csr_matrix.
        """

        if X.shape[1] != Z.shape[1]:
            raise ValueError(
                f"X and Z should have same dimension to do convex combination. {X.shape[1]}!={X.shape[1]}."
            )
        if isinstance(alpha, np.ndarray):
            if alpha.ndim != 1:
                raise ValueError(f"If alpha is a numpy array, it should be an 1-d array")

            if alpha.shape[0] != Z.shape[0]:
                raise ValueError(
                    f"If alpha is a numpy array, the length should be equal to the number of samples."
                )

            if np.any(alpha < 0.0) or np.any(alpha > 1.0):
                raise ValueError(f"All values in alpha should be between 0 and 1.")
        elif isinstance(alpha, (int, float)):
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError(f"alpha should be between 0 and 1.")
        else:
            raise NotImplementedError("alpha should be a number or a numpy array")

        pifa = LabelEmbeddingFactory.pifa(Y, X, threads=threads, normalized_Y=normalized_Y)

        if isinstance(alpha, (int, float)):
            alpha = np.repeat(alpha, Z.shape[0])

        alpha_complementary = 1.0 - alpha

        if not isinstance(pifa, (smat.csr_matrix, np.ndarray)) or not isinstance(
            Z, (smat.csr_matrix, np.ndarray)
        ):
            raise NotImplementedError(
                "type(pifa) and type(Z) should be row-major spmatrix or ndarray"
            )
        if isinstance(pifa, smat.csr_matrix) and isinstance(Z, smat.csr_matrix):
            label_embedding = smat_util.csr_rowwise_mul(pifa, alpha) + smat_util.csr_rowwise_mul(
                Z, alpha_complementary
            )
        else:
            if isinstance(pifa, smat.csr_matrix):
                pifa = pifa.toarray()
            if isinstance(Z, smat.csr_matrix):
                Z = Z.toarray()
            label_embedding = alpha[:, None] * pifa + alpha_complementary[:, None] * Z

        return label_embedding
