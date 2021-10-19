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
import collections

import numpy as np
import scipy.sparse as smat


def cs_matrix(arg1, mat_type, shape=None, dtype=None, copy=False, check_contents=False):
    """Custom compressed sparse matrix constructor that allows indices and indptr to be stored in different types.

    Args:
        arg1 (tuple): (data, indices, indptr) to construct compressed sparse matrix
        mat_type (type): the matrix type to construct, one of [scipy.sparse.csr_matrix | scipy.sparse.csc_matrix]
        shape (tuple, optional): shape of the matrix, default None to infer from arg1
        dtype (type, optional): type of values in the matrix, default None to infer from data
        copy (bool, optional): whether to copy the input arrays, defaults to False
        check_contents (bool, optional): whether to check array contents to determine dtype, defaults to False

    Returns:
        compressed sparse matrix in mat_type
    """
    (data, indices, indptr) = arg1
    indices_dtype = smat.sputils.get_index_dtype(indices, check_contents=check_contents)
    indptr_dtype = smat.sputils.get_index_dtype(indptr, check_contents=check_contents)

    ret = mat_type(shape, dtype=dtype)
    # Read matrix dimensions given, if any
    if shape is None:
        # shape not already set, try to infer dimensions
        try:
            major_dim = len(ret.indptr) - 1
            minor_dim = ret.indices.max() + 1
        except Exception:
            raise ValueError("unable to infer matrix dimensions")
        else:
            shape = ret._swap((major_dim, minor_dim))

    ret.indices = np.array(indices, copy=copy, dtype=indices_dtype)
    ret.indptr = np.array(indptr, copy=copy, dtype=indptr_dtype)
    ret.data = np.array(data, copy=copy, dtype=dtype)

    return ret


def csr_matrix(arg1, shape=None, dtype=None, copy=False):
    """Custom csr_matrix constructor that allows indices and indptr to be stored in different types.

    Args:
        arg1 (tuple): (data, indices, indptr) to construct csr_matrix
        shape (tuple, optional): shape of the matrix, default None to infer from arg1
        dtype (type, optional): type of values in the matrix, default None to infer from data
        copy (bool, optional): whether to copy the input arrays, defaults to False

    Returns:
        csr_matrix
    """
    return cs_matrix(arg1, smat.csr_matrix, shape=shape, dtype=dtype, copy=copy)


def csc_matrix(arg1, shape=None, dtype=None, copy=False):
    """Custom csc_matrix constructor that allows indices and indptr to be stored in different types.

    Args:
        arg1 (tuple): (data, indices, indptr) to construct csc_matrix
        shape (tuple, optional): shape of the matrix, default None to infer from arg1
        dtype (type, optional): type of values in the matrix, default None to infer from data
        copy (bool, optional): whether to copy the input arrays, defaults to False

    Returns:
        csc_matrix
    """
    return cs_matrix(arg1, smat.csc_matrix, shape=shape, dtype=dtype, copy=copy)


def save_matrix(tgt, mat):
    """Save dense or sparse matrix to file.

    Args:
        tgt (str): path to save the matrix
        mat (numpy.ndarray or scipy.sparse.spmatrix): target matrix to save
    """
    assert isinstance(tgt, str), "tgt for save_matrix must be a str, but got {}".format(type(tgt))
    with open(tgt, "wb") as tgt_file:
        if isinstance(mat, np.ndarray):
            np.save(tgt_file, mat, allow_pickle=False)
        elif isinstance(mat, smat.spmatrix):
            smat.save_npz(tgt_file, mat, compressed=False)
        else:
            raise NotImplementedError("Save not implemented for matrix type {}".format(type(mat)))


def load_matrix(src, dtype=None):
    """Load dense or sparse matrix from file.

    Args:
        src (str): path to load the matrix.
        dtype (numpy.dtype, optional): if given, convert matrix dtype. otherwise use default type.

    Returns:
        mat (numpy.ndarray or scipy.sparse.spmatrix): loaded matrix

    Notes:
        If underlying matrix is {"csc", "csr", "bsr"}, indices will be sorted.
    """
    if not isinstance(src, str):
        raise ValueError("src for load_matrix must be a str")

    mat = np.load(src)
    # decide whether it's dense or sparse
    if isinstance(mat, np.ndarray):
        pass
    elif isinstance(mat, np.lib.npyio.NpzFile):
        # Ref code: https://github.com/scipy/scipy/blob/v1.4.1/scipy/sparse/_matrix_io.py#L19-L80
        matrix_format = mat["format"].item()
        if not isinstance(matrix_format, str):
            # files saved with SciPy < 1.0.0 may contain unicode or bytes.
            matrix_format = matrix_format.decode("ascii")
        try:
            cls = getattr(smat, "{}_matrix".format(matrix_format))
        except AttributeError:
            raise ValueError("Unknown matrix format {}".format(matrix_format))

        if matrix_format in ("csc", "csr", "bsr"):
            mat = cls((mat["data"], mat["indices"], mat["indptr"]), shape=mat["shape"])
            # This is in-place operation
            mat.sort_indices()
        elif matrix_format == "dia":
            mat = cls((mat["data"], mat["offsets"]), shape=mat["shape"])
        elif matrix_format == "coo":
            mat = cls((mat["data"], (mat["row"], mat["col"])), shape=mat["shape"])
        else:
            raise NotImplementedError(
                "Load is not implemented for sparse matrix of format {}.".format(matrix_format)
            )
    else:
        raise TypeError("load_feature_matrix encountered unknown input format {}".format(type(mat)))

    if dtype is None:
        return mat
    else:
        return mat.astype(dtype)


def transpose(mat):
    """Transpose a dense/sparse matrix.

    Args:
        X (np.ndarray, spmatrix): input matrix to be transposed.

    Returns:
        transposed X
    """

    if not isinstance(mat, smat.spmatrix):
        raise ValueError("mat must be a smat.spmatrix type")

    if isinstance(mat, smat.csr_matrix):
        return csc_matrix((mat.data, mat.indices, mat.indptr), shape=(mat.shape[1], mat.shape[0]))
    elif isinstance(mat, smat.csc_matrix):
        return csr_matrix((mat.data, mat.indices, mat.indptr), shape=(mat.shape[1], mat.shape[0]))
    else:
        return mat.T


def sorted_csr_from_coo(shape, row_idx, col_idx, val, only_topk=None):
    """Return a row-sorted CSR matrix from a COO sparse matrix.

    Nonzero elements in each row of the returned CSR matrix is sorted in an descending order based on the value. If only_topk is given, only topk largest elements will be kept.

    Args:
        shape (tuple): the shape of the input COO matrix
        row_idx (ndarray): row indices of the input COO matrix
        col_idx (ndarray): col indices of the input COO matrix
        val (ndarray): values of the input COO matrix
        only_topk (int, optional): keep only topk elements per row. Default None to ignore

    Returns:
        csr_matrix
    """
    csr = smat.csr_matrix((val, (row_idx, col_idx)), shape=shape)
    csr.sort_indices()
    for i in range(shape[0]):
        rng = slice(csr.indptr[i], csr.indptr[i + 1])
        sorted_idx = np.argsort(-csr.data[rng], kind="mergesort")
        csr.indices[rng] = csr.indices[rng][sorted_idx]
        csr.data[rng] = csr.data[rng][sorted_idx]
    if only_topk is not None:
        assert isinstance(only_topk, int), f"Wrong type: type(only_topk) = {type(only_topk)}"
        only_topk = max(min(1, only_topk), only_topk)
        nnz_of_insts = csr.indptr[1:] - csr.indptr[:-1]
        row_idx = np.repeat(np.arange(shape[0], dtype=csr.indices.dtype), nnz_of_insts)
        selected_idx = (np.arange(len(csr.data)) - csr.indptr[row_idx]) < only_topk
        row_idx = row_idx[selected_idx]
        col_idx = csr.indices[selected_idx]
        val = csr.data[selected_idx]
        indptr = np.cumsum(np.bincount(row_idx + 1, minlength=(shape[0] + 1)))
        csr = csr_matrix((val, col_idx, indptr), shape=shape, dtype=val.dtype)
    return csr


def sorted_csc_from_coo(shape, row_idx, col_idx, val, only_topk=None):
    """Return a column-sorted CSC matrix from a COO sparse matrix.

    Nonzero elements in each col of the returned CSC matrix is sorted in an descending order based on the value. If only_topk is given, only topk largest elements will be kept.

    Args:
        shape (tuple): the shape of the input COO matrix
        row_idx (ndarray): row indices of the input COO matrix
        col_idx (ndarray): col indices of the input COO matrix
        val (ndarray): values of the input COO matrix
        only_topk (int, optional): keep only topk elements per col. Default None to ignore

    Returns:
        csc_matrix
    """
    csr = sorted_csr_from_coo(shape[::-1], col_idx, row_idx, val, only_topk=None)
    return transpose(csr)


def binarized(X, inplace=False):
    """Binarize a dense/sparse matrix. All nonzero elements become 1.

    Args:
        X (np.ndarray, spmatrix): input matrix to binarize
        inplace (bool, optional): if True do the binarization in-place, else return a copy. Default False

    Returns:
        binarized X
    """

    if not isinstance(X, (np.ndarray, smat.spmatrix)):
        raise NotImplementedError(
            "this function only support X being np.ndarray or scipy.sparse.spmatrix."
        )

    if not inplace:
        X = X.copy()

    if isinstance(X, smat.spmatrix):
        X.data[:] = 1
    else:
        X[:] = 1

    return X


def sorted_csr(csr, only_topk=None):
    """Return a copy of input CSR matrix where nonzero elements in each row is sorted in an descending order based on the value.

    If `only_topk` is given, only top-k largest elements will be kept.

    Args:
        csr (csr_matrix): input csr_matrix to sort
        only_topk (int, optional): keep only topk elements per row. Default None to ignore

    Returns:
        csr_matrix
    """
    if not isinstance(csr, smat.csr_matrix):
        raise ValueError("the input matrix must be a csr_matrix.")

    row_idx = np.repeat(np.arange(csr.shape[0], dtype=np.uint32), csr.indptr[1:] - csr.indptr[:-1])
    return sorted_csr_from_coo(csr.shape, row_idx, csr.indices, csr.data, only_topk)


def sorted_csc(csc, only_topk=None):
    """Return a copy of input CSC matrix where nonzero elements in each column is sorted in an descending order based on the value.

    If `only_topk` is given, only top-k largest elements will be kept.

    Args:
        csc (csc_matrix): input csc_matrix to sort
        only_topk (int, optional): keep only topk elements per col. Default None to ignore

    Returns:
        csc_matrix
    """
    if not isinstance(csc, smat.csc_matrix):
        raise ValueError("the input matrix must be a csc_matrix.")

    return transpose(sorted_csr(transpose(csc)))


def dense_to_csr(dense, topk=None, batch=None):
    """Memory efficient method to construct a csr_matrix from a dense matrix.

    Args:
        dense (ndarray): 2-D dense matrix to convert.
        topk (int or None, optional): keep topk non-zeros with largest abs value for each row.
             Default None to keep everything.
        batch (int or None, optional): the batch size for construction.
             Default None to use min(dense.shape[0], 10 ** 5).

    Returns:
        csr_matrix that has topk nnz each row with the same shape as dense.
    """

    BATCH_LIMIT = 10 ** 5

    if topk is None:
        keep_topk = dense.shape[1]
    else:
        keep_topk = min(dense.shape[1], max(1, int(topk)))

    # if batch is given, use input batch size even if input batch > BATCH_LIMIT
    if batch is None:
        chunk_size = min(dense.shape[0], BATCH_LIMIT)
    else:
        chunk_size = min(dense.shape[0], max(1, int(batch)))

    max_nnz = keep_topk * dense.shape[0]
    indptr_dtype = np.int32 if max_nnz < np.iinfo(np.int32).max else np.int64
    indices_dtype = np.int32 if dense.shape[1] < np.iinfo(np.int32).max else np.int64

    data = np.empty((keep_topk * dense.shape[0],), dtype=dense.dtype)
    indices = np.empty((keep_topk * dense.shape[0],), dtype=indices_dtype)
    for i in range(0, dense.shape[0], chunk_size):
        cur_chunk = dense[i : i + chunk_size, :]
        chunk_len = cur_chunk.shape[0]
        if keep_topk < dense.shape[1]:
            col_indices = np.argpartition(abs(cur_chunk), keep_topk, axis=1)[:, -keep_topk:]
        else:
            col_indices = np.repeat(np.arange(keep_topk)[np.newaxis, :], chunk_len, axis=0)
        row_indices = np.repeat(np.arange(chunk_len)[:, np.newaxis], keep_topk, axis=1)
        chunk_data = cur_chunk[row_indices, col_indices]

        data[i * keep_topk : i * keep_topk + chunk_data.size] = chunk_data.flatten()
        indices[i * keep_topk : i * keep_topk + col_indices.size] = col_indices.flatten()
    indptr = np.arange(0, dense.shape[0] * keep_topk + 1, keep_topk, dtype=indptr_dtype)
    # Bypass scipy constructor to allow different indices and indptr types
    return csr_matrix((data, indices, indptr), shape=dense.shape)


def vstack_csr(matrices, dtype=None):
    """Memory efficient method to stack csr_matrices vertically.

    The returned matrix will retain the indices order.

    Args:
        matrices (list or tuple of csr_matrix): the matrices to stack in order, with shape (M1 x N), (M2 x N), ...
        dtype (dtype, optional): The data-type of the output matrix. Default None to infer from matrices

    Returns:
        csr_matrix with shape (M1 + M2 + ..., N)
    """
    if not isinstance(matrices, (list, tuple)):
        raise ValueError("matrices should be either list or tuple")
    if any(not isinstance(X, smat.csr_matrix) for X in matrices):
        raise ValueError("all matrix in matrices need to be csr_matrix!")
    if len(matrices) <= 1:
        return matrices[0] if len(matrices) == 1 else None
    nr_cols = matrices[0].shape[1]
    if any(mat.shape[1] != nr_cols for mat in matrices):
        raise ValueError("Second dim not match")

    total_nnz = sum([int(mat.nnz) for mat in matrices])
    total_rows = sum([int(mat.shape[0]) for mat in matrices])

    # infer result dtypes from inputs
    int32max = np.iinfo(np.int32).max
    if dtype is None:
        dtype = smat.sputils.upcast(*[mat.dtype for mat in matrices])
    indices_dtype = np.int64 if nr_cols > int32max else np.int32
    indptr_dtype = np.int64 if total_nnz > int32max else np.int32

    indptr = np.empty(total_rows + 1, dtype=indptr_dtype)
    indices = np.empty(total_nnz, dtype=indices_dtype)
    data = np.empty(total_nnz, dtype=dtype)

    indptr[0], cur_nnz, cur_row = 0, 0, 0
    for mat in matrices:
        indices[cur_nnz : cur_nnz + mat.nnz] = mat.indices
        data[cur_nnz : cur_nnz + mat.nnz] = mat.data
        # can not merge the following two lines because
        # mat.indptr[1:] + cur_nnz may overflow!
        indptr[cur_row + 1 : cur_row + mat.shape[0] + 1] = mat.indptr[1:]
        indptr[cur_row + 1 : cur_row + mat.shape[0] + 1] += cur_nnz
        cur_nnz += mat.nnz
        cur_row += mat.shape[0]

    return csr_matrix((data, indices, indptr), shape=(total_rows, nr_cols))


def hstack_csr(matrices, dtype=None):
    """Memory efficient method to stack csr_matrices horizontally.

    The returned matrix will retain the indices order.

    Args:
        matrices (list or tuple of csr_matrix): the matrices to stack in order, with shape (M x N1), (M x N2), ...
        dtype (dtype, optional): The data-type of the output matrix. Default None to infer from matrices

    Returns:
        csr_matrix with shape (M, N1 + N2 + ...)
    """
    if not isinstance(matrices, (list, tuple)):
        raise ValueError("matrices should be either list or tuple")
    if any(not isinstance(X, smat.csr_matrix) for X in matrices):
        raise ValueError("all matrix in matrices need to be csr_matrix!")
    if len(matrices) <= 1:
        return matrices[0] if len(matrices) == 1 else None
    nr_rows = matrices[0].shape[0]
    if any(mat.shape[0] != nr_rows for mat in matrices):
        raise ValueError("First dim not match")

    total_nnz = sum([int(mat.nnz) for mat in matrices])
    total_cols = sum([int(mat.shape[1]) for mat in matrices])
    # infer result dtypes from inputs
    int32max = np.iinfo(np.int32).max
    if dtype is None:
        dtype = smat.sputils.upcast(*[mat.dtype for mat in matrices])
    indices_dtype = np.int64 if nr_rows > int32max else np.int32
    indptr_dtype = np.int64 if total_nnz > int32max else np.int32

    indptr = np.empty(nr_rows + 1, dtype=indptr_dtype)
    indices = np.empty(total_nnz, dtype=indices_dtype)
    data = np.empty(total_nnz, dtype=dtype)
    indptr[0], cur_ptr = 0, 0
    for i in range(nr_rows):  # for every row
        start_col = 0
        for mat in matrices:
            cur_nnz = mat.indptr[i + 1] - mat.indptr[i]
            indices[cur_ptr : cur_ptr + cur_nnz] = (
                mat.indices[mat.indptr[i] : mat.indptr[i + 1]] + start_col
            )
            data[cur_ptr : cur_ptr + cur_nnz] = mat.data[mat.indptr[i] : mat.indptr[i + 1]]
            cur_ptr += cur_nnz
            start_col += mat.shape[1]
        indptr[i + 1] = cur_ptr

    return csr_matrix((data, indices, indptr), shape=(nr_rows, total_cols))


def block_diag_csr(matrices, dtype=None):
    """Memory efficient method to stack csr_matrices block diagonally.

    The returned matrix will retain the indices order.

    Args:
        matrices (list or tuple of csr_matrix): the matrices to stack in order, with shape (NR1 x NC1), (NR2 x NC2), ...
        dtype (dtype, optional): The data-type of the output matrix. Default None to infer from matrices

    Returns:
        csr_matrix with shape (NR1 + NR2 + ..., NC1 + NC2 + ...)
    """
    if not isinstance(matrices, (list, tuple)):
        raise ValueError("matrices should be either list or tuple")
    if any(not isinstance(X, smat.csr_matrix) for X in matrices):
        raise ValueError("all matrix in matrices need to be csr_matrix!")
    if len(matrices) <= 1:
        return matrices[0] if len(matrices) == 1 else None

    total_nnz = sum([int(mat.nnz) for mat in matrices])
    total_rows = sum([int(mat.shape[0]) for mat in matrices])
    total_cols = sum([int(mat.shape[1]) for mat in matrices])
    # infer result dtypes from inputs
    int32max = np.iinfo(np.int32).max
    if dtype is None:
        dtype = smat.sputils.upcast(*[mat.dtype for mat in matrices])
    indices_dtype = np.int64 if total_rows > int32max else np.int32
    indptr_dtype = np.int64 if total_nnz > int32max else np.int32

    indptr = np.empty(total_rows + 1, dtype=indptr_dtype)
    indices = np.empty(total_nnz, dtype=indices_dtype)
    data = np.empty(total_nnz, dtype=dtype)
    cur_row, cur_col, cur_nnz = 0, 0, 0
    indptr[0] = 0
    for mat in matrices:
        data[cur_nnz : cur_nnz + mat.nnz] = mat.data
        indices[cur_nnz : cur_nnz + mat.nnz] = mat.indices + cur_col
        indptr[1 + cur_row : 1 + cur_row + mat.shape[0]] = mat.indptr[1:] + indptr[cur_row]
        cur_col += mat.shape[1]
        cur_row += mat.shape[0]
        cur_nnz += mat.nnz
    return csr_matrix((data, indices, indptr), shape=(total_rows, total_cols))


def vstack_csc(matrices, dtype=None):
    """Memory efficient method to stack csc_matrices vertically.

    The returned matrix will retain the indices order.

    Args:
        matrices (list or tuple of csc_matrix): the matrices to stack in order, with shape (M1 x N), (M2 x N), ...
        dtype (dtype, optional): The data-type of the output matrix. Default None to infer from matrices

    Returns:
        csc_matrix with shape (M1 + M2 + ..., N)
    """
    if not isinstance(matrices, (list, tuple)):
        raise ValueError("matrices should be either list or tuple")
    if any(not isinstance(X, smat.csc_matrix) for X in matrices):
        raise ValueError("all matrix in matrices need to be csc_matrix!")

    if len(matrices) <= 1:
        return matrices[0] if len(matrices) == 1 else None
    return transpose(hstack_csr([transpose(mat) for mat in matrices], dtype=dtype))


def hstack_csc(matrices, dtype=None):
    """Memory efficient method to stack csc_matrices horizontally.

    The returned matrix will retain the indices order.

    Args:
        matrices (list or tuple of csc_matrix): the matrices to stack in order, with shape (M x N1), (M x N2), ...
        dtype (dtype, optional): The data-type of the output matrix. Default None to infer from matrices

    Returns:
        csc_matrix with shape (M, N1 + N2 + ...)
    """
    if not isinstance(matrices, (list, tuple)):
        raise ValueError("matrices should be either list or tuple")
    if any(not isinstance(X, smat.csc_matrix) for X in matrices):
        raise ValueError("all matrix in matrices need to be csc_matrix!")

    if len(matrices) <= 1:
        return matrices[0] if len(matrices) == 1 else None
    return transpose(vstack_csr([transpose(mat) for mat in matrices], dtype=dtype))


def block_diag_csc(matrices, dtype=None):
    """Memory efficient method to stack csc_matrices block diagonally.

    The returned matrix will retain the indices order.

    Args:
        matrices (list or tuple of csr_matrix): the matrices to stack in order, with shape (NR1 x NC1), (NR2 x NC2), ...
        dtype (dtype, optional): The data-type of the output matrix. Default None to infer from matrices

    Returns:
        csc_matrix with shape (NR1+ NR2 + ..., NC1 + NC2 + ...)
    """
    if not isinstance(matrices, (list, tuple)):
        raise ValueError("matrices should be either list or tuple")
    if any(not isinstance(X, smat.csc_matrix) for X in matrices):
        raise ValueError("all matrix in matrices need to be csc_matrix!")

    if len(matrices) <= 1:
        return matrices[0] if len(matrices) == 1 else None
    return transpose(block_diag_csr([transpose(mat) for mat in matrices], dtype=dtype))


def get_csc_col_nonzero(matrix):
    """Given a matrix, returns the nonzero row ids of each col

    The returned ndarray will retain the indices order.

    Args:
        matrix: the matrix to operate on, with shape (N x M)

    Returns:
        list of ndarray [a_1, a_2, a_3, ...], where a_i is an array indicate the nonzero row ids of col i
    """
    if not isinstance(matrix, smat.csc_matrix):
        raise ValueError("matrix need to be csc_matrix!")
    return [matrix.indices[matrix.indptr[i] : matrix.indptr[i + 1]] for i in range(matrix.shape[1])]


def get_csr_row_nonzero(matrix):
    """Given a matrix, returns the nonzero col ids of each row

    The returned ndarray will retain the indices order.

    Args:
        matrix: the matrix to operate on, with shape (N x M)

    Returns:
        list of ndarray [a_1, a_2, a_3, ...], where a_i is an array indicate the nonzero col ids of row i
    """
    if not isinstance(matrix, smat.csr_matrix):
        raise ValueError("matrix need to be csr_matrix!")
    return [matrix.indices[matrix.indptr[i] : matrix.indptr[i + 1]] for i in range(matrix.shape[0])]


def get_row_submatrices(matrices, row_indices):
    """Get the sub-matrices of given matrices by selecting the rows given in row_indices

    Args:
        matrices (list of csr_matrix or ndarray): the matrices [mat_1, mat_2, ...] to operate on, with shape (M x N1), (M x N2), ...
        row_indices (list or ndarray): the row indices to select

    Returns:
        list of csr_matrix or ndarray

    """
    if not isinstance(matrices, (list, tuple)):
        raise ValueError("matrices should be either list or tuple")
    n_mat = len(matrices)
    if n_mat == 0:
        raise ValueError("At least one matrix required as input")

    if any(not isinstance(X, (smat.csr_matrix, np.ndarray)) for X in matrices):
        raise ValueError("all matrix in matrices need to be csr_matrix or ndarray!")
    nr_rows = matrices[0].shape[0]
    if any(mat.shape[0] != nr_rows for mat in matrices):
        raise ValueError("First dim not match")
    if any(idx >= nr_rows or idx < 0 for idx in row_indices):
        raise ValueError("row indices should be positive and do not exceed matrix first dimension")

    results = []
    for mat in matrices:
        mat1 = mat[row_indices, :]
        if isinstance(mat, smat.csr_matrix):
            mat1.sort_indices()
        results += [mat1]

    return results


def dense_to_coo(dense):
    """Convert a dense matrix to COO format.

    Args:
        dense (ndarray): input dense matrix

    Returns:
        coo_matrix
    """
    rows = np.arange(dense.shape[0], dtype=np.uint32)
    cols = np.arange(dense.shape[1], dtype=np.uint32)
    row_idx = np.repeat(rows, np.ones_like(rows) * len(cols)).astype(np.uint32)
    col_idx = np.ones((len(rows), 1), dtype=np.uint32).dot(cols.reshape(1, -1)).ravel()
    return smat.coo_matrix((dense.ravel(), (row_idx, col_idx)), shape=dense.shape)


def get_relevance_csr(csr, mm=None, dtype=np.float64):
    """Return the csr matrix containing relevance scores based on given prediction csr matrix.

    Relevance score is defined as: max_rank - local_rank + 1

    Args:
        csr (csr_matrix): input CSR matrix, row indices are sorted in descending order
        mm (int, optional): max rank, will be inferred from csr if not given
        dtype (type, optional): datatype for the returned relevance matrix. Default float64.

    Returns:
        csr_matrix of relevance scores
    """
    if mm is None:
        mm = (csr.indptr[1:] - csr.indptr[:-1]).max()
    nnz = len(csr.data)
    nnz_of_rows = csr.indptr[1:] - csr.indptr[:-1]
    row_idx = np.repeat(np.arange(csr.shape[0]), nnz_of_rows)
    rel = np.array(
        mm - (np.arange(nnz) - csr.indptr[row_idx]), dtype=dtype
    )  # adding 1 to avoiding zero entries
    return smat.csr_matrix((rel, csr.indices, csr.indptr), csr.shape)


def get_sparsified_coo(coo, selected_rows, selected_columns):
    """
    Zero out everything not in selected rows and columns.

    Args:
        coo (coo_matrix): input coo matrix
        selected_rows (list of int or np.array(int)): list of rows to be not zeroed out
        selected_columns (list of int or np.array(int)): list of columns to be not zeroed out

    Returns:
        coo matrix with unwanted rows and columns zeroed out.
    """
    valid_rows = np.zeros(coo.shape[0], dtype=bool)
    valid_cols = np.zeros(coo.shape[1], dtype=bool)
    valid_rows[selected_rows] = True
    valid_cols[selected_columns] = True
    valid_idx = valid_rows[coo.row] & valid_cols[coo.col]
    coo = smat.coo_matrix(
        (coo.data[valid_idx], (coo.row[valid_idx], coo.col[valid_idx])), shape=coo.shape
    )
    return coo


def csr_rowwise_mul(A, v):
    """Row-wise multiplication between sparse csr matrix A and dense array v.

    Where each row of A is multiplied by the corresponding element in v.
    The number of rows of A is same as the length of v.

    Args:
        A (csr_matrix): The matrix to be multiplied.
        v (ndarray): The multiplying vector.

    Returns:
        Z (csr_matrix): The product of row-wise multiplication of A and v.
    """

    if not isinstance(A, smat.csr_matrix):
        raise ValueError(f"A must be scipy.sparse.csr_matrix")
    if not isinstance(v, np.ndarray):
        raise ValueError(f"v must be a numpy ndarray")
    if v.ndim != 1:
        raise ValueError(f"v should be an 1-d array")
    if v.shape[0] != A.shape[0]:
        raise ValueError(f"The dimension of v should be the same as the number of rows of A")

    Z = A.copy()
    for i in range(v.shape[0]):
        Z.data[Z.indptr[i] : Z.indptr[i + 1]] *= v[i]
    return Z


def csc_colwise_mul(A, v):
    """Column-wise multiplication between sparse csc matrix A and dense array v, where each column of A is multiplied by the corresponding element in v (The number of columns of A is same as the length of v).

    Args:
        A (csc_matrix): The matrix to be multiplied.
        v (ndarray): The multiplying vector.

    Returns:
        Z (csc_matrix): The product of column-wise multiplication of A and v.
    """

    if not isinstance(A, smat.csc_matrix):
        raise ValueError(f"A must be scipy.sparse.csc_matrix")
    if not isinstance(v, np.ndarray):
        raise ValueError(f"v must be a numpy ndarray")
    if v.ndim != 1:
        raise ValueError(f"v should be an 1-d array")
    if v.shape[0] != A.shape[1]:
        raise ValueError(f"The dimension of v should be the same as the number of columns of A")

    Z = A.copy()
    for i in range(v.shape[0]):
        Z.data[Z.indptr[i] : Z.indptr[i + 1]] *= v[i]
    return Z


def get_cocluster_spectral_embeddings(A, dim=24):
    """Obtain the co-cluster spectral embeddings for the given bipartite graph described in [1]

    * [1] `Dhillon, Inderjit S, 2001. Co-clustering documents and words using
          bipartite spectral graph partition`

    Args:
        A (csr_matrix or csc_matrix): bipartite graph matrix
        dim (int, optional): the dimension of the returned embeddings. Default 24

    Returns:
        (row_embedding, col_embedding): a tuple of embeddings for rows and columns respectively
            row_embedding: numpy.ndarray of shape (A.shape[0], dim).
            col_embedding: numpy.ndarray of shape (A.shape[1], dim).
    """
    assert A.min() >= 0.0, "A must be nonnegative"

    from sklearn.utils.extmath import randomized_svd

    # Obtain An, the normalized adjacency bipartite matrix described in Eq (10) of [1]
    #   A_n = D_1^{-1/2} A D_2^{-1/2}
    #   row_diag = diagonal of D_1^{-1/2}
    #   col_diag = diagonal of D_2^{-1/2}
    row_diag = np.asarray(np.sqrt(A.sum(axis=1))).squeeze()
    col_diag = np.asarray(np.sqrt(A.sum(axis=0))).squeeze()
    row_diag[row_diag == 0] = 1.0
    col_diag[col_diag == 0] = 1.0
    row_diag = 1.0 / row_diag
    col_diag = 1.0 / col_diag
    if smat.issparse(A):
        n_rows, n_cols = A.shape
        r = smat.dia_matrix((row_diag, [0]), shape=(n_rows, n_rows))
        c = smat.dia_matrix((col_diag, [0]), shape=(n_cols, n_cols))
        An = r * A * c
    else:
        An = row_diag[:, np.newaxis] * A * col_diag

    # run SVD on An
    nr_discards = 1  # discarding the first component
    U, Sigma, VT = randomized_svd(An, dim + nr_discards, random_state=0)

    # Normalized the singular vectors based on Eq (24) of [1]
    row_embedding = np.ascontiguousarray(row_diag[:, np.newaxis] * U[:, nr_discards:])
    col_embedding = np.ascontiguousarray(col_diag[:, np.newaxis] * VT[nr_discards:].T)

    return row_embedding, col_embedding


def csr_row_softmax(mat, inplace=False):
    """Apply row-wise softmax transform to csr_matrix

    Args:
        mat (csr_matrix): input csr_matrix to transform
        inplace (bool, optional): if True do the transform in-place, else return a copy. Default False

    Returns:
        row-wise softmaxed mat
    """
    if not isinstance(mat, smat.csr_matrix):
        raise ValueError(f"Got {type(mat)} when expecting csr_matrix")

    from scipy.special import softmax

    if not inplace:
        mat = mat.copy()

    for i in range(mat.shape[0]):
        rng = slice(mat.indptr[i], mat.indptr[i + 1])
        mat.data[rng] = softmax(mat.data[rng])
    return mat


class CsrEnsembler(object):
    """A class implementing several ensemblers for a list sorted CSR predictions"""

    @staticmethod
    def check_validlity(*args):
        """Check whether input CSR matrices are valid

        Args:
            args (iterable over csr_matrix): input CSR matrices
        """
        for x in args:
            assert isinstance(x, smat.csr_matrix), type(x)
        assert all(x.shape == args[0].shape for x in args)

    @staticmethod
    def average(*args):
        """Ensemble predictions by averaging prediction values

        Args:
            args (iterable over csr_matrix): input CSR matrices

        Returns:
            ret (csr_matrix): ensembled prediction CSR matrix
        """
        CsrEnsembler.check_validlity(*args)
        ret = sum(args)
        ret = sorted_csr(ret)
        ret.data /= len(args)
        return ret

    @staticmethod
    def rank_average(*args):
        """Ensemble predictions by averaging prediction ranks

        Args:
            args (iterable over csr_matrix): input CSR matrices

        Returns:
            ret (csr_matrix): ensembled prediction CSR matrix
        """
        CsrEnsembler.check_validlity(*args)
        mm = max((x.indptr[1:] - x.indptr[:-1]).max() for x in args)
        ret = sum(get_relevance_csr(csr, mm) for csr in args)
        ret = sorted_csr(ret)
        ret.data /= len(args)
        return ret

    @staticmethod
    def sigmoid_average(*args):
        """Ensemble predictions by averaging sigmoid transformed prediction values

        Args:
            args (iterable over csr_matrix): input CSR matrices

        Returns:
            ret (csr_matrix): ensembled prediction CSR matrix
        """
        CsrEnsembler.check_validlity(*args)

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        for i in range(len(args)):
            args[i].data = sigmoid(args[i].data)
        ret = sum(args)
        ret = sorted_csr(ret)
        ret.data /= len(args)
        return ret

    @staticmethod
    def softmax_average(*args):
        """Ensemble predictions by averaging softmax normalized prediction values
        The softmax is only applied on non-zero entrees of the matrix.

        Args:
            args (iterable over csr_matrix): input CSR matrices

        Returns:
            ret (csr_matrix): ensembled prediction CSR matrix
        """
        CsrEnsembler.check_validlity(*args)
        # apply softmax
        args = [csr_row_softmax(x) for x in args]
        ret = sum(args)
        ret = sorted_csr(ret)
        ret.data /= len(args)
        return ret

    @staticmethod
    def round_robin(*args):
        """Ensemble predictions by round robin

        Args:
            args (iterable over csr_matrix): input CSR matrices

        Returns:
            ret (csr_matrix): ensembled prediction CSR matrix
        """
        CsrEnsembler.check_validlity(*args)
        base = 1.0 / (len(args) + 1.0)
        mm = max((x.indptr[1:] - x.indptr[:-1]).max() for x in args)
        ret = get_relevance_csr(args[0], mm)
        ret.data[:] += len(args) * base
        for i, x in enumerate(args[1:], 1):
            tmp = get_relevance_csr(x, mm)
            tmp.data[:] += (len(args) - i) * base
            ret = ret.maximum(tmp)
        ret = sorted_csr(ret)
        ret.data /= len(args)
        return ret

    @staticmethod
    def print_ens(Ytrue, pred_set, param_set, ens_method="rank_average", topk=10):
        """Print metrics before and after ensemble

        Args:
            Ytrue (csr_matrix): ground truth label matrix
            pred_set (iterable over csr_matrix): prediction matrices to ensemble
            param_set (iterable): parameters or model names associated with pred_set
            ens_method (list or str): list of ensemble methods or single str. Default 'rank_average'
            topk (int, optional): only generate topk prediction. Default 10
        """

        for param, pred in zip(param_set, pred_set):
            print("param: {}".format(param))
            print(Metrics.generate(Ytrue, pred, topk=topk))

        if not isinstance(ens_method, list):
            ens_method = [ens_method]
        for ens_name in ens_method:
            ens = getattr(CsrEnsembler, ens_name)
            cur_pred = ens(*pred_set)
            print(f"==== {ens_name} ensemble results ====")
            print(Metrics.generate(Ytrue, cur_pred, topk=topk))


class Metrics(collections.namedtuple("Metrics", ["prec", "recall"])):
    """The metrics (precision, recall) for multi-label classification problems."""

    __slots__ = ()

    def __str__(self):
        """Format printing"""

        def fmt(key):
            return " ".join("{:4.2f}".format(100 * v) for v in getattr(self, key)[:])

        return "\n".join("{:7}= {}".format(key, fmt(key)) for key in self._fields)

    @classmethod
    def default(cls):
        """Default dummy metric"""
        return cls(prec=[], recall=[])

    @classmethod
    def generate(cls, tY, pY, topk=10):
        """Compute the metrics with given prediction and ground truth.

        Args:
            tY (csr_matrix): ground truth label matrix
            pY (csr_matrix): predicted logits
            topk (int, optional): only generate topk prediction. Default 10

        Returns:
            Metrics
        """
        assert isinstance(tY, smat.csr_matrix), type(tY)
        assert isinstance(pY, smat.csr_matrix), type(pY)
        assert tY.shape == pY.shape, "tY.shape = {}, pY.shape = {}".format(tY.shape, pY.shape)
        pY = sorted_csr(pY)
        total_matched = np.zeros(topk, dtype=np.uint64)
        recall = np.zeros(topk, dtype=np.float64)
        for i in range(tY.shape[0]):
            truth = tY.indices[tY.indptr[i] : tY.indptr[i + 1]]
            matched = np.isin(pY.indices[pY.indptr[i] : pY.indptr[i + 1]][:topk], truth)
            cum_matched = np.cumsum(matched, dtype=np.uint64)
            total_matched[: len(cum_matched)] += cum_matched
            recall[: len(cum_matched)] += cum_matched / max(len(truth), 1)
            if len(cum_matched) != 0:
                total_matched[len(cum_matched) :] += cum_matched[-1]
                recall[len(cum_matched) :] += cum_matched[-1] / max(len(truth), 1)
        prec = total_matched / tY.shape[0] / np.arange(1, topk + 1)
        recall = recall / tY.shape[0]
        return cls(prec=prec, recall=recall)
