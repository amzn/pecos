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
from pytest import approx


def test_importable():
    import pecos.xmc  # noqa: F401
    from pecos import xmc  # noqa: F401
    from pecos.xmc import xlinear  # noqa: F401
    from pecos.xmc import xtransformer  # noqa: F401
    from pecos.xmc import Indexer  # noqa: F401


def test_hierarchicalkmeans():
    import numpy as np
    import scipy.sparse as smat
    from sklearn.preprocessing import normalize
    from pecos.xmc import Indexer

    feat_mat = normalize(
        smat.csr_matrix([[1, 0], [0.95, 0.05], [0.9, 0.1], [0, 1]], dtype=np.float32)
    )
    target_balanced = [0, 0, 1, 1]

    balanced_chain = Indexer.gen(feat_mat, max_leaf_size=3)
    balanced_assignments = (balanced_chain[-1].todense() == [0, 1]).all(axis=1).A1
    assert np.array_equal(balanced_assignments, target_balanced) or np.array_equal(
        ~balanced_assignments, target_balanced
    )

    chain2 = Indexer.gen(feat_mat, max_leaf_size=1, nr_splits=2)
    chain4 = Indexer.gen(feat_mat, max_leaf_size=1, nr_splits=4)

    assert (chain2.chain[-1] - chain4.chain[-1]).nnz == 0

    assert (chain2.chain[1].dot(chain2.chain[0]) - chain4.chain[0]).nnz == 0


def test_hierarchicalkmeans_sampling():
    import numpy as np
    import scipy.sparse as smat
    from sklearn.preprocessing import normalize
    from pecos.xmc import Indexer
    from pecos.xmc.base import HierarchicalKMeans

    # randomly sampling arbitary number of examples from the 4 following instances results in the same clustering results.
    feat_mat = normalize(
        smat.csr_matrix([[1, 0], [0.99, 0.02], [0.01, 1.03], [0, 1]], dtype=np.float32)
    )
    target_balanced = [0, 0, 1, 1]

    # the clustering results are the same as long as min_sample_rate >= 0.25
    train_params = HierarchicalKMeans.TrainParams(
        do_sample=True,
        min_sample_rate=0.75,
        warmup_ratio=1.0,
        max_leaf_size=2,
    )
    balanced_chain = Indexer.gen(feat_mat, train_params=train_params)
    balanced_assignments = (balanced_chain[-1].todense() == [0, 1]).all(axis=1).A1
    assert np.array_equal(balanced_assignments, target_balanced) or np.array_equal(
        ~balanced_assignments, target_balanced
    )


def test_label_embedding():
    import random
    import numpy as np
    import scipy.sparse as smat
    from sklearn.preprocessing import normalize
    from pecos.xmc import LabelEmbeddingFactory

    X = smat.csr_matrix(smat.eye(3)).astype(np.float32)
    X_dense = X.toarray()
    Y = np.array([[1, 1, 1, 1, 0], [1, 1, 0, 1, 1], [0, 1, 1, 1, 1]])
    Y = smat.csr_matrix(Y).astype(np.float32)
    Lt_dense = np.array(
        [
            [0.70710678, 0.70710678, 0.0],
            [0.57735027, 0.57735027, 0.57735027],
            [0.70710678, 0.0, 0.70710678],
            [0.57735027, 0.57735027, 0.57735027],
            [0.0, 0.70710678, 0.70710678],
        ]
    )
    Lt = smat.csr_matrix(Lt_dense)

    # pifa, X.dtype = csr_matrix, and simple X/Y with closed-form Lt_dense
    Lp = LabelEmbeddingFactory.create(Y, X, method="pifa").toarray()
    assert Lt_dense == approx(
        Lp, abs=1e-6
    ), f"Lt_dense (true label embedding) != Lp (pifa label embedding), where closed-form X is sparse"

    # pifa, X.dtype = np.array, and the same X/Y with previous closed-form Lt_dense
    Lp = LabelEmbeddingFactory.create(Y, X_dense, method="pifa")
    assert Lt_dense == approx(
        Lp, abs=1e-6
    ), f"Lt_dense (true label embedding) != Lp (pifa label embedding), where closed-form X is dense"

    # test data for pifa_lf_concat and pifa_lf_convex_combine
    Lp = LabelEmbeddingFactory.create(Y, X_dense, method="pifa")
    Lt_half_dense = Lt_dense * 0.5
    Lt_half = smat.csr_matrix(Lt_half_dense)

    # test data for pifa_lf_concat
    Lplc_true = np.hstack([Lp, Lt_half_dense])

    # pifa_lf_concat, X.dtype = ndarray, Z.dtype = ndarray
    Lplc = LabelEmbeddingFactory.create(Y, X_dense, Z=Lt_half_dense, method="pifa_lf_concat")
    assert isinstance(
        Lplc, np.ndarray
    ), f"Return matrix should be np.ndarray when X.dtype = ndarray, Z.dtype = ndarray"
    assert Lplc == approx(
        Lplc_true
    ), f"Lplc_true (true label embedding) != Lplc (pifa_lf_concat label embedding), where X.dtype = ndarray, Z.dtype = ndarray"

    # pifa_lf_concat, X.dtype = ndarray, Z.dtype = csr_matrix
    Lplc = LabelEmbeddingFactory.create(Y, X_dense, Z=Lt_half, method="pifa_lf_concat")
    assert isinstance(
        Lplc, smat.csr_matrix
    ), f"Return matrix should be csr_matrix when X.dtype = ndarray, Z.dtype = csr_matrix"
    assert Lplc.toarray() == approx(
        Lplc_true
    ), f"Lplc_true (true label embedding) != Lplc (pifa_lf_concat label embedding), where X.dtype = ndarray, Z.dtype = csr_matrix"

    # pifa_lf_concat, X.dtype = csr_matrix, Z.dtype = ndarray
    Lplc = LabelEmbeddingFactory.create(Y, X, Z=Lt_half_dense, method="pifa_lf_concat")
    assert isinstance(
        Lplc, smat.csr_matrix
    ), f"Return matrix should be csr_matrix when X.dtype = csr_matrix, Z.dtype = ndarray"
    assert Lplc.toarray() == approx(
        Lplc_true
    ), f"Lplc_true (true label embedding) != Lplc (pifa_lf_concat label embedding), where X.dtype = csr_matrix, Z.dtype = ndarray"

    # pifa_lf_concat, X.dtype = csr_matrix, Z.dtype = csr_matrix
    Lplc = LabelEmbeddingFactory.create(Y, X, Z=Lt_half, method="pifa_lf_concat")
    assert isinstance(
        Lplc, smat.csr_matrix
    ), f"Return matrix should be csr_matrix when X.dtype = csr_matrix, Z.dtype = csr_matrix"
    assert Lplc.toarray() == approx(
        Lplc_true
    ), f"Lplc_true (true label embedding) != Lplc (pifa_lf_concat label embedding), where X.dtype = csr_matrix, Z.dtype = csr_matrix"

    # pifa_lf_convex_combine, alpha is a number
    alpha = 0.3
    Lplcvx_true = alpha * Lp + (1.0 - alpha) * Lt_half_dense

    # pifa_lf_convex_combine, X.dtype = ndarray, Z.dtype = ndarray
    Lplcvx = LabelEmbeddingFactory.create(
        Y, X_dense, Z=Lt_half_dense, alpha=alpha, method="pifa_lf_convex_combine"
    )
    assert isinstance(
        Lplcvx, np.ndarray
    ), f"Return matrix should be ndarray when X.dtype = ndarray, Z.dtype = ndarray"
    assert Lplcvx == approx(
        Lplcvx_true
    ), f"Lplcvx_true (true label embedding) != Lplcvx (pifa_lf_convex_combine label embedding), where X.dtype = ndarray, Z.dtype = ndarray"

    # pifa_lf_convex_combine, X.dtype = ndarray, Z.dtype = csr_matrix
    Lplcvx = LabelEmbeddingFactory.create(
        Y, X_dense, Z=Lt_half, alpha=alpha, method="pifa_lf_convex_combine"
    )
    assert isinstance(
        Lplcvx, np.ndarray
    ), f"Return matrix should be ndarray when X.dtype = ndarray, Z.dtype = csr_matrix"
    assert Lplcvx == approx(
        Lplcvx_true
    ), f"Lplcvx_true (true label embedding) != Lplcvx (pifa_lf_convex_combine label embedding), where X.dtype = ndarray, Z.dtype = csr_matrix"

    # pifa_lf_convex_combine, X.dtype = csr_matrix, Z.dtype = ndarray
    Lplcvx = LabelEmbeddingFactory.create(
        Y, X, Z=Lt_half_dense, alpha=alpha, method="pifa_lf_convex_combine"
    )
    assert isinstance(
        Lplcvx, np.ndarray
    ), f"Return matrix should be ndarray when X.dtype = csr_matrix, Z.dtype = ndarray"
    assert Lplcvx == approx(
        Lplcvx_true
    ), f"Lplcvx_true (true label embedding) != Lplcvx (pifa_lf_convex_combine label embedding), where X.dtype = csr_matrix, Z.dtype = ndarray"

    # pifa_lf_convex_combine, X.dtype = csr_matrix, Z.dtype = csr_matrix
    Lplcvx = LabelEmbeddingFactory.create(
        Y, X, Z=Lt_half, alpha=alpha, method="pifa_lf_convex_combine"
    )
    assert isinstance(
        Lplcvx, smat.csr_matrix
    ), f"Return matrix should be csr_matrix when X.dtype = csr_matrix, Z.dtype = csr_matrix"
    assert Lplcvx.toarray() == approx(
        Lplcvx_true
    ), f"Lplcvx_true (true label embedding) != Lplcvx (pifa_lf_convex_combine label embedding), where X.dtype = csr_matrix, Z.dtype = csr_matrix"

    # pifa_lf_convex_combine, alpha is an 1-D array
    alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    Lplcvx_true = np.zeros_like(Lp)
    for i in range(Lp.shape[0]):
        Lplcvx_true[i, :] = alpha[i] * Lp[i, :] + (1.0 - alpha[i]) * Lt_half_dense[i, :]

    # pifa_lf_convex_combine, X.dtype = ndarray, Z.dtype = ndarray
    Lplcvx = LabelEmbeddingFactory.create(
        Y, X_dense, Z=Lt_half_dense, alpha=alpha, method="pifa_lf_convex_combine"
    )
    assert isinstance(
        Lplcvx, np.ndarray
    ), f"Return matrix should be ndarray when X.dtype = ndarray, Z.dtype = ndarray"
    assert Lplcvx == approx(
        Lplcvx_true
    ), f"Lplcvx_true (true label embedding) != Lplcvx (pifa_lf_convex_combine label embedding), where X.dtype = ndarray, Z.dtype = ndarray"

    # pifa_lf_convex_combine, X.dtype = ndarray, Z.dtype = csr_matrix
    Lplcvx = LabelEmbeddingFactory.create(
        Y, X_dense, Z=Lt_half, alpha=alpha, method="pifa_lf_convex_combine"
    )
    assert isinstance(
        Lplcvx, np.ndarray
    ), f"Return matrix should be ndarray when X.dtype = ndarray, Z.dtype = csr_matrix"
    assert Lplcvx == approx(
        Lplcvx_true
    ), f"Lplcvx_true (true label embedding) != Lplcvx (pifa_lf_convex_combine label embedding), where X.dtype = ndarray, Z.dtype = csr_matrix"

    # pifa_lf_convex_combine, X.dtype = csr_matrix, Z.dtype = ndarray
    Lplcvx = LabelEmbeddingFactory.create(
        Y, X, Z=Lt_half_dense, alpha=alpha, method="pifa_lf_convex_combine"
    )
    assert isinstance(
        Lplcvx, np.ndarray
    ), f"Return matrix should be ndarray when X.dtype = csr_matrix, Z.dtype = ndarray"
    assert Lplcvx == approx(
        Lplcvx_true
    ), f"Lplcvx_true (true label embedding) != Lplcvx (pifa_lf_convex_combine label embedding), where X.dtype = csr_matrix, Z.dtype = ndarray"

    # pifa_lf_convex_combine, X.dtype = csr_matrix, Z.dtype = csr_matrix
    Lplcvx = LabelEmbeddingFactory.create(
        Y, X, Z=Lt_half, alpha=alpha, method="pifa_lf_convex_combine"
    )
    assert isinstance(
        Lplcvx, smat.csr_matrix
    ), f"Return matrix should be csr_matrix when X.dtype = csr_matrix, Z.dtype = csr_matrix"
    assert Lplcvx.toarray() == approx(
        Lplcvx_true
    ), f"Lplcvx_true (true label embedding) != Lplcvx (pifa_lf_convex_combine label embedding), where X.dtype = csr_matrix, Z.dtype = csr_matrix"

    # pifa, X.dtype = csr_matrix, and random sampling X/Y
    np.random.seed(1234)
    random.seed(1234)
    N = 30
    D = 40
    Z = 50
    p = 0.05
    X = smat.random(N, D, density=p, format="csr").astype(np.float32)
    Y = smat.random(N, Z, density=p, format="csr").astype(np.float32)
    Y_avg = normalize(Y, axis=1, norm="l2").tocsc()
    Lt = normalize(X.T.dot(Y_avg).T, axis=1, norm="l2", copy=False).toarray()
    Lp = LabelEmbeddingFactory.create(Y, X, method="pifa").toarray()
    assert Lt == approx(
        Lp, abs=1e-6
    ), f"Lt (true label embedding) != Lp (pifa label embedding), where random X is sparse"

    # pii, Y.dtype = csr_matrix
    Y = np.array([[1, 1, 1, 1, 0], [1, 1, 0, 1, 1], [0, 1, 1, 1, 1]])
    Y = smat.csr_matrix(Y).astype(np.float32)
    Lp = LabelEmbeddingFactory.create(Y, method="pii")
    Lt = normalize(Y.T.tocsr(copy=True), axis=1, norm="l2", copy=False)
    assert Lt.todense() == approx(
        Lp.todense(), abs=1e-6
    ), f"Lt (true label embedding) != Lp (pii label embedding)"
