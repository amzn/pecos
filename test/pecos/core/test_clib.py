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
    import pecos.core  # noqa: F401
    from pecos.core import clib  # noqa: F401


def test_smat_matmul():
    import numpy as np
    import scipy.sparse as smat
    from pecos.core import clib

    X = smat.csr_matrix([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
    Y = smat.csr_matrix([[0.5, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    gt_XYT = np.array(
        [[0.50, 0.00, 1.00, 0.00], [0.25, 0.50, 0.50, 0.25], [0.00, 1.00, 0.00, 0.50]],
        dtype=np.float32,
    )

    pd_XYT = clib.sparse_matmul(X, Y.T, threads=2).toarray()

    assert gt_XYT == approx(
        pd_XYT, abs=1e-9
    ), f"gt_XYT (true matmul) != pd_XYT (our pecos.sparse_matmul), where X/Y are sparse"


def test_sparse_inner_products():
    import numpy as np
    import scipy.sparse as smat
    from pecos.core import clib

    X = smat.csr_matrix([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
    Y = smat.csr_matrix([[0.5, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    gt_XYT = np.array(
        [[0.50, 0.00, 1.00, 0.00], [0.25, 0.50, 0.50, 0.25], [0.00, 1.00, 0.00, 0.50]],
        dtype=np.float32,
    )
    X_row_idx = np.array([0, 1, 2], dtype=np.uint32)
    Y_col_idx = np.array([1, 2, 3], dtype=np.uint32)
    true_vals = np.array(
        [gt_XYT[i, j] for i, j in zip(X_row_idx, Y_col_idx)],
        dtype=np.float32,
    )

    # test csr2csc
    pred_vals = clib.sparse_inner_products(X, Y.T, X_row_idx, Y_col_idx)
    assert true_vals == approx(
        pred_vals, abs=1e-9
    ), f"true_vals != pred_vals, where X/Y are csr/csc"

    # test drm2dcm
    X_dense = X.toarray()
    Y_dense = Y.toarray()
    pred_vals = clib.sparse_inner_products(X_dense, Y_dense.T, X_row_idx, Y_col_idx)
    assert true_vals == approx(
        pred_vals, abs=1e-9
    ), f"true_vals != pred_vals, where X/Y are drm/dcm"


def test_platt_scale():
    import numpy as np
    from pecos.core import clib

    A = 0.25
    B = 3.14

    orig = np.arange(-15, 15, 1, dtype=np.float32)
    tgt = np.array([1.0 / (1 + np.exp(A * t + B)) for t in orig], dtype=np.float32)
    At, Bt = clib.fit_platt_transform(orig, tgt)
    assert B == approx(Bt, abs=1e-6), f"Platt_scale B error: {B} != {Bt}"
    assert A == approx(At, abs=1e-6), f"Platt_scale A error: {A} != {At}"

    orig = np.arange(-15, 15, 1, dtype=np.float64)
    tgt = np.array([1.0 / (1 + np.exp(A * t + B)) for t in orig], dtype=np.float64)
    At, Bt = clib.fit_platt_transform(orig, tgt)
    assert B == approx(Bt, abs=1e-6), f"Platt_scale B error: {B} != {Bt}"
    assert A == approx(At, abs=1e-6), f"Platt_scale A error: {A} != {At}"
