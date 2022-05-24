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


def test_save_load_matrix(tmpdir):
    from pecos.utils import smat_util
    import numpy as np
    import scipy.sparse as smat

    A = smat.csr_matrix([[0, 1, 0, 1], [3, 0, 3, 0], [1, 0, 0, 1], [3, 0, 3, 0]])
    A_dir = tmpdir.join("A").realpath().strpath
    smat_util.save_matrix(A_dir, A)
    A_load = smat_util.load_matrix(A_dir)
    assert isinstance(A, smat.spmatrix)
    assert A.todense() == approx(A_load.todense(), abs=1e-6)
    # dense case
    B = A.toarray()
    B_dir = tmpdir.join("B").realpath().strpath
    smat_util.save_matrix(B_dir, B)
    B_load = smat_util.load_matrix(B_dir)
    assert isinstance(B_load, np.ndarray)
    assert B == approx(B_load, abs=1e-6)


def test_get_sparsified_coo():
    from pecos.utils import smat_util
    from scipy import sparse as smat

    data = [[1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0]]
    coo = smat.coo_matrix(data)
    selected_rows = [1, 2]
    selected_columns = [0, 2]
    new_coo = smat_util.get_sparsified_coo(coo, selected_rows, selected_columns).toarray()
    for i in range(new_coo.shape[0]):
        for j in range(new_coo.shape[1]):
            if i in selected_rows and j in selected_columns:
                assert new_coo[i, j] == data[i][j]
            else:
                assert new_coo[i, j] == 0


def test_get_cocluster_spectral_embeddings():
    from pecos.utils import smat_util
    import numpy as np
    from scipy import sparse as smat

    A = smat.csr_matrix([[0, 1, 0, 1], [3, 0, 3, 0], [1, 0, 0, 1], [3, 0, 3, 0]])
    expected_row_embedding = np.array([[0.60162737], [-0.13128586], [0.18608776], [-0.13128586]])
    expected_col_embedding = np.array([[-0.09449112], [0.66143783], [-0.14433757], [0.4330127]])

    row_embedding, col_embedding = smat_util.get_cocluster_spectral_embeddings(A, dim=1)

    assert row_embedding == approx(expected_row_embedding, abs=1e-6)
    assert col_embedding == approx(expected_col_embedding, abs=1e-6)


def test_dense_to_csr():
    from pecos.utils import smat_util
    import numpy as np

    X = np.array([[-5.0, 1.0, 2.0, 10.0], [-4.0, 2.0, 0.0, 1.0], [-10.0, 11.0, 2.0, 1.0]])
    X_csr = smat_util.dense_to_csr(X, topk=2, batch=2)
    X_res = np.array([[-5.0, 0.0, 0.0, 10.0], [-4.0, 2.0, 0.0, 0.0], [-10.0, 11.0, 0.0, 0.0]])
    assert X_csr.todense() == approx(X_res)
    X_csr = smat_util.dense_to_csr(X, topk=1000, batch=1000)
    assert X_csr.todense() == approx(X)


def test_stack_csr():
    from pecos.utils import smat_util
    from scipy import sparse as smat
    import numpy as np

    X0 = np.array([[-5.0, 1.0, 0.0, 10.0], [0.0, 2.0, 0.0, 1.0], [-10.0, 11.0, 2.0, 0.0]])
    X1 = smat.csr_matrix(X0)
    X2 = smat.csr_matrix(X0)
    X_hstack = smat_util.hstack_csr([X1, X2])
    assert X_hstack.todense() == approx(np.hstack([X0, X0]))
    assert X_hstack.dtype == X1.dtype
    assert type(X_hstack) == smat.csr_matrix
    X_vstack = smat_util.vstack_csr([X1, X2])
    assert X_vstack.todense() == approx(np.vstack([X0, X0]))
    assert X_vstack.dtype == X1.dtype
    assert type(X_vstack) == smat.csr_matrix
    X_block_diag = smat_util.block_diag_csr([X1, X2])
    X_np_block_diag = np.hstack(
        [np.vstack([X0, np.zeros_like(X0)]), np.vstack([np.zeros_like(X0), X0])]
    )
    assert X_block_diag.todense() == approx(X_np_block_diag)
    assert X_block_diag.dtype == X1.dtype
    assert type(X_block_diag) == smat.csr_matrix


def test_stack_csc():
    from pecos.utils import smat_util
    from scipy import sparse as smat
    import numpy as np

    X0 = np.array([[-5.0, 1.0, 0.0, 10.0], [0.0, 2.0, 0.0, 1.0], [-10.0, 11.0, 2.0, 0.0]])
    X1 = smat.csc_matrix(X0)
    X2 = smat.csc_matrix(X0)
    X_hstack = smat_util.hstack_csc([X1, X2])
    assert X_hstack.todense() == approx(np.hstack([X0, X0]))
    assert X_hstack.dtype == X1.dtype
    assert type(X_hstack) == smat.csc_matrix
    X_vstack = smat_util.vstack_csc([X1, X2])
    assert X_vstack.todense() == approx(np.vstack([X0, X0]))
    assert X_vstack.dtype == X1.dtype
    assert type(X_vstack) == smat.csc_matrix
    X_block_diag = smat_util.block_diag_csc([X1, X2])
    X_np_block_diag = np.hstack(
        [np.vstack([X0, np.zeros_like(X0)]), np.vstack([np.zeros_like(X0), X0])]
    )
    assert X_block_diag.todense() == approx(X_np_block_diag)
    assert X_block_diag.dtype == X1.dtype
    assert type(X_block_diag) == smat.csc_matrix


def test_get_col_row_nonzero():
    from pecos.utils import smat_util
    from scipy import sparse as smat
    import numpy as np

    X = np.array([[-5.0, 1.0, 0.0, 10.0], [0.0, 2.0, 0.0, 1.0], [-10.0, 11.0, 2.0, 0.0]])
    X_csr = smat.csr_matrix(X)
    X_csr.sort_indices()
    row_nonzero = smat_util.get_csr_row_nonzero(X_csr)
    X_csc = smat.csc_matrix(X)
    X_csc.sort_indices()
    col_nonzero = smat_util.get_csc_col_nonzero(X_csc)
    np_row_nonzero = [[] for _ in range(X.shape[0])]
    np_col_nonzero = [[] for _ in range(X.shape[1])]
    xs, ys = np.nonzero(X)
    for x, y in zip(xs, ys):
        np_row_nonzero[x].append(y)
        np_col_nonzero[y].append(x)
    np_row_nonzero = [np.array(ys) for ys in np_row_nonzero]
    np_col_nonzero = [np.array(xs) for xs in np_col_nonzero]

    assert len(np_row_nonzero) == len(row_nonzero)
    for i in range(len(row_nonzero)):
        assert len(row_nonzero) == len(np_row_nonzero)
        for ys, np_ys in zip(row_nonzero, np_row_nonzero):
            assert np.all(ys == np_ys)

    assert len(np_col_nonzero) == len(col_nonzero)
    for i in range(len(col_nonzero)):
        assert len(col_nonzero) == len(np_col_nonzero)
        for xs, np_xs in zip(col_nonzero, np_col_nonzero):
            assert np.all(xs == np_xs)


def test_csr_rowwise_mul():
    from pecos.utils import smat_util
    from scipy import sparse as smat
    import numpy as np

    X0 = np.array([[-5.0, 1.0, 0.0, 10.0], [0.0, 2.0, 0.0, 1.0], [-10.0, 11.0, 2.0, 0.0]])
    X1 = smat.csr_matrix(X0)
    v = np.array([3.0, 4.0, 5.0])
    prod = smat_util.csr_rowwise_mul(X1, v)
    assert isinstance(prod, smat.csr_matrix)
    assert prod.todense() == approx(X0 * v[:, None])


def test_csc_colwise_mul():
    from pecos.utils import smat_util
    from scipy import sparse as smat
    import numpy as np

    X0 = np.array([[-5.0, 1.0, 0.0, 10.0], [0.0, 2.0, 0.0, 1.0], [-10.0, 11.0, 2.0, 0.0]])
    X1 = smat.csc_matrix(X0)
    v = np.array([3.0, 4.0, 5.0, 6.0])
    prod = smat_util.csc_colwise_mul(X1, v)
    assert isinstance(prod, smat.csc_matrix)
    assert prod.todense() == approx(X0 * v[None, :])


def test_get_row_submatrices():
    from pecos.utils import smat_util
    import numpy as np
    import scipy.sparse as smat

    row_indices = [0, 2, 1]
    rows = []

    rows += [-5.0, 1.0]
    rows += [0.0, 2.0]
    rows += [2.0, 0.0]
    rows += [1.2, 0.0]

    X0 = np.vstack(rows)
    X1 = smat.csr_matrix(X0)
    Xres = np.vstack([rows[i] for i in row_indices])

    X0_sub, X1_sub = smat_util.get_row_submatrices([X0, X1], row_indices)

    assert type(X0_sub) == type(X0)
    assert X0_sub == approx(Xres)
    assert type(X1_sub) == type(X1)
    assert X1_sub.todense() == approx(Xres)


def test_csr_ensembler():
    from pecos.utils import smat_util
    from scipy import sparse as smat
    import numpy as np

    X0 = np.array([[0.5, 0, 0, 1.0], [0, 0.2, 0, 0.5]])
    X1 = np.array([[1.0, 0, 0, 0.2], [0, 0, 0.2, 0]])
    X0 = smat_util.sorted_csr(smat.csr_matrix(X0))
    X1 = smat_util.sorted_csr(smat.csr_matrix(X1))

    # average
    X_avr = np.array([[0.75, 0, 0, 0.6], [0, 0.1, 0.1, 0.25]])
    X_avr_pred = smat_util.CsrEnsembler.average(X0, X1).todense()
    assert X_avr_pred == approx(X_avr), X_avr_pred

    # rank_average
    X_rank = np.array([[1.5, 0, 0, 1.5], [0, 0.5, 1, 1]])
    X_rank_pred = smat_util.CsrEnsembler.rank_average(X0, X1).todense()
    assert X_rank_pred == approx(X_rank), X_rank_pred

    # sigmoid_average
    X_sig = np.array([[0.67675895, 0.0, 0.0, 0.64044629], [0.0, 0.274917, 0.274917, 0.31122967]])
    X_sig_pred = smat_util.CsrEnsembler.sigmoid_average(X0, X1).todense()
    assert X_sig_pred == approx(X_sig), X_sig_pred

    # softmax_average
    X_soft = np.array([[0.5090297, 0, 0, 0.4909703], [0, 0.24092582, 0.5, 0.25907418]])
    X_soft_pred = smat_util.CsrEnsembler.softmax_average(X0, X1).todense()
    assert X_soft_pred == approx(X_soft), X_soft_pred

    # round_robin
    X_rr = np.array([[1.16666667, 0, 0, 1.33333333], [0, 0.83333333, 1.16666667, 1.33333333]])
    X_rr_pred = smat_util.CsrEnsembler.round_robin(X0, X1).todense()
    assert X_rr_pred == approx(X_rr), X_rr_pred


def test_sorted_csr():
    from pecos.utils import smat_util
    from scipy import sparse as smat
    import numpy as np

    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    X1 = smat.csr_matrix((data, indices, indptr), shape=(3, 3))

    sorted_data = np.array([2, 1, 3, 6, 5, 4])
    sorted_indices = np.array([2, 0, 2, 2, 1, 0], dtype=int)

    sorted_X1 = smat_util.sorted_csr(X1)
    assert isinstance(sorted_X1, smat.csr_matrix)
    assert sorted_X1.todense() == approx(X1.todense())
    assert sorted_X1.data == approx(sorted_data)
    assert sorted_X1.indices == approx(sorted_indices)


def test_sorted_csc():
    from pecos.utils import smat_util
    from scipy import sparse as smat
    import numpy as np

    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    X1 = smat.csc_matrix((data, indices, indptr), shape=(3, 3))

    sorted_data = np.array([2, 1, 3, 6, 5, 4])
    sorted_indices = np.array([2, 0, 2, 2, 1, 0], dtype=int)

    sorted_X1 = smat_util.sorted_csc(X1)
    assert isinstance(sorted_X1, smat.csc_matrix)
    assert sorted_X1.todense() == approx(X1.todense())
    assert sorted_X1.data == approx(sorted_data)
    assert sorted_X1.indices == approx(sorted_indices)


def test_csr_row_softmax():
    from pecos.utils import smat_util
    from scipy import sparse as smat
    import numpy as np

    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    X0 = smat.csr_matrix((data, indices, indptr), shape=(3, 3))

    softmaxed_X0 = smat_util.csr_row_softmax(X0).todense()

    softmaxed_mat = np.array(
        [[0.2689414214, 0, 0.7310585786], [0, 0, 1], [0.0900305732, 0.2447284711, 0.6652409558]]
    )

    assert softmaxed_X0 == approx(softmaxed_mat)
