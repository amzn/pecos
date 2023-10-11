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


def test_num_mmap_valstore(tmpdir):
    from pecos.utils.mmap_valstore_util import MmapValStore, MmapValStoreBatchGetter
    import numpy as np

    store_dir = tmpdir.join("num_valstore").realpath().strpath

    #  [[ 0.,  1.,  2.],
    #   [ 3.,  4.,  5.],
    #   [ 6.,  7.,  8.],
    #   [ 9., 10., 11.],
    #   [12., 13., 14.]]
    arr = np.arange(15, dtype=np.float32).reshape(5, 3)

    # Write-only Mode
    w_store = MmapValStore("float32")
    w_store.open("w", store_dir)
    # from array
    w_store.store.from_vals(arr)
    # Size
    assert w_store.store.n_row(), w_store.store.n_col() == arr.shape
    w_store.close()

    # Read-only Mode
    r_store = MmapValStore("float32")
    r_store.open("r", store_dir)
    # Get sub-matrix
    vs_getter = MmapValStoreBatchGetter(r_store.store, max_row_size=10, max_col_size=10)
    assert np.array(vs_getter.get([0], [0])).tolist() == [[0.0]]
    assert np.array(vs_getter.get([0, 1], [0, 1])).tolist() == [[0.0, 1.0], [3.0, 4.0]]
    assert np.array(vs_getter.get([2, 3, 4], [1])).tolist() == [[7.0], [10.0], [13.0]]
    assert np.array(vs_getter.get([3, 1], [0, 2])).tolist() == [[9.0, 11.0], [3.0, 5.0]]
    assert np.array(vs_getter.get([4], [1, 0, 2])).tolist() == [[13.0, 12.0, 14.0]]


def test_str_mmap_valstore(tmpdir):
    from pecos.utils.mmap_valstore_util import MmapValStore, MmapValStoreBatchGetter

    store_dir = tmpdir.join("str_valstore").realpath().strpath

    # [['0', '00', '000'],
    #  ['1', '11', '111'],
    #  ['2', '22', '222'],
    #  ['3', '33', '333'],
    #  ['4', '44', '444']]
    n_row = 5
    n_col = 3
    str_list = [[f"{j}" * (i + 1) for i in range(n_col)] for j in range(n_row)]
    flat_str_list = [item for sublist in str_list for item in sublist]

    # Write-only Mode
    w_store = MmapValStore("str")
    w_store.open("w", store_dir)
    # from array
    w_store.store.from_vals((n_row, n_col, flat_str_list))
    # Size
    assert (w_store.store.n_row(), w_store.store.n_col()) == (n_row, n_col)
    w_store.close()

    # Read-only Mode
    r_store = MmapValStore("str")
    r_store.open("r", store_dir)
    # Get sub-matrix
    vs_getter = MmapValStoreBatchGetter(
        r_store.store, max_row_size=10, max_col_size=10, trunc_val_len=10
    )

    sub_rows, sub_cols = [3, 1], [0, 2]
    str_sub_mat = vs_getter.get(sub_rows, sub_cols)
    for i in range(len(sub_rows)):
        for j in range(len(sub_cols)):
            assert str_sub_mat[i][j] == str_list[sub_rows[i]][sub_cols[j]]  # noqa: W503

    sub_rows, sub_cols = [4, 4, 1, 2], [1, 2, 0]
    str_sub_mat = vs_getter.get(sub_rows, sub_cols)
    for i in range(len(sub_rows)):
        for j in range(len(sub_cols)):
            assert str_sub_mat[i][j] == str_list[sub_rows[i]][sub_cols[j]]  # noqa: W503
