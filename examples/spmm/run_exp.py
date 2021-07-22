
import argparse
import os
import time
import numpy as np
import scipy.sparse as smat
from pecos.utils import smat_util
from pecos.core import clib as pecos_clib

SPMM_ALGO_LIST = ["pecos", "tensorflow", "pytorch", "intel-mkl", "scipy"]


def csr_to_coo(A):
    A_coo = smat.coo_matrix(A)
    # (nnz, 2)
    indices = np.vstack([A_coo.row, A_coo.col]).T
    # (nnz, )
    values = A_coo.data
    return indices, values


def do_spmm_exp(args):
    # load data
    Y = smat_util.load_matrix(args.y_npz_path).astype(np.float32)
    X = smat_util.load_matrix(args.x_npz_path).astype(np.float32)
    YT_csr = Y.T.tocsr()
    X_csr = X.tocsr()

    # The #threads is control by env variables (except for pecos)
    # e.g., export OMP_NUM_THREADS=16, export MKL_NUM_THREADS=16.
    run_time = 0.0
    if args.spmm_algo == "pecos":
        start = time.time()
        Z = pecos_clib.sparse_matmul(
            YT_csr, X_csr,
            eliminate_zeros=False,
            sorted_indices=True,
            threads=args.threads,
        )
        run_time += time.time() - start
        Z_data = Z.data
    elif args.spmm_algo == "intel-mkl":
        from sparse_dot_mkl import dot_product_mkl
        # make sure set the index to int64 for large matrices
        # export MKL_INTERFACE_LAYER=ILP64
        start = time.time()
        Z = dot_product_mkl(YT_csr, X_csr, reorder_output=True)
        run_time += time.time() - start
        Z_data = Z.data
    elif args.spmm_algo == "scipy":
        # scipy will not sorted the indices for each row,
        # so we do it explicitly
        start = time.time()
        Z = YT_csr.dot(X_csr)
        Z.sort_indices()
        run_time += time.time() - start
        Z_data = Z.data
    elif args.spmm_algo == "pytorch":
        import torch
        def get_pt_data(A_csr):
            A_indices, A_values = csr_to_coo(A_csr)
            A_pt = torch.sparse_coo_tensor(
                A_indices.T.astype(np.int64),
                A_values.astype(np.float32),
                A_csr.shape,
            )
            return A_pt
        YT_pt = get_pt_data(YT_csr)
        X_pt = get_pt_data(X_csr)
        start = time.time()
        Z_pt = torch.sparse.mm(YT_pt, X_pt)
        run_time += time.time() - start
        Z_data = Z_pt.coalesce().values().numpy()
    elif args.spmm_algo == "tensorflow":
        import tensorflow.compat.v1 as tf
        from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
        def get_tf_data(A_csr):
            # Define (COO format) Sparse Tensors over Numpy arrays
            A_indices, A_values = csr_to_coo(A_csr)
            A_st = tf.sparse.SparseTensor(
                A_indices.astype(np.int64),
                A_values.astype(np.float32),
                A_csr.shape,
            )
            return A_st
        # Tensorflow (v2.5.0) usage, as of 07/20/2021:
        # https://www.tensorflow.org/api_docs/python/tf/raw_ops/SparseMatrixSparseMatMul
        with tf.Session() as sess:
            YT_st = get_tf_data(YT_csr)
            X_st = get_tf_data(X_csr)
            sess.run(YT_st)
            sess.run(X_st)
            YT_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(YT_st.indices, YT_st.values, YT_st.dense_shape)
            X_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(X_st.indices, X_st.values, X_st.dense_shape)
            start = time.time()
            Z_sm = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(a=YT_sm, b=X_sm, type=tf.float32)
            Z_st = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(Z_sm, tf.float32)
            Z_data = sess.run(Z_st.values)
            run_time += time.time() - start
    else:
        raise ValueError(f"spmm_algo={args.spmm_algo} is not valid")
    run_time = time.time() - start
    print("algo {:16s} time(s) {:9.5f} nnz(Z) {:12d} mu(Z.data) {:8.4f}".format(
        args.spmm_algo,
        run_time,
        len(Z_data),
        np.mean(Z_data),
        )
    )


def parse_arguments():
    """Parse arguments"""

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "-x", "--x-npz-path",
        type=str, required=True,
        help="path to the CSR npz of the sparse instance-to-feature matrix",
    )
    parser.add_argument(
        "-y", "--y-npz-path",
        type=str, required=True,
        help="path to the CSR npz of the sparse instance-to-label matrix",
    )
    # Optional
    parser.add_argument(
        "-algo", "--spmm-algo",
        choices=SPMM_ALGO_LIST,
        type=str, default="pecos",
        help=f"SpMM algorithm (default pecos). Available choices are {', '.join(SPMM_ALGO_LIST)}",
    )
    parser.add_argument(
        "-t", "--threads",
        type=int, default=-1,
        help=f"number of threads for spmm_algo=pecos (default -1 to use all)",
    )

    return parser


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    do_spmm_exp(args)
