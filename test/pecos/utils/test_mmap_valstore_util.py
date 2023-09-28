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
    from pecos.utils.mmap_valstore_util import MmapValStore, MmapValStoreSubMatGetter
    import numpy as np

    map_dir = tmpdir.join("num_valstore").realpath().strpath

    #  [[ 0.,  1.,  2.],
    #   [ 3.,  4.,  5.],
    #   [ 6.,  7.,  8.],
    #   [ 9., 10., 11.],
    #   [12., 13., 14.]]
    arr = np.arange(15, dtype=np.float32).reshape(5, 3)

    # Write-only Mode
    w_store = MmapValStore("num_f32")
    w_store.open("w", map_dir)
    # from array
    w_store.store.from_vals(arr)
    # Size
    assert w_store.store.n_row(), w_store.store.n_col() == arr.shape
    w_store.close()

    # Read-only Mode
    r_store = MmapValStore("num_f32")
    r_store.open("r", map_dir)
    # Get sub-matrix
    vs_getter = MmapValStoreSubMatGetter(r_store.store, max_row_size=10, max_col_size=10)
    assert np.array(vs_getter.get_submat([0], [0])).tolist() == [[0.0]]
    assert np.array(vs_getter.get_submat([0, 1], [0, 1])).tolist() == [[0.0, 1.0], [3.0, 4.0]]
    assert np.array(vs_getter.get_submat([2, 3, 4], [1])).tolist() == [[7.0], [10.0], [13.0]]
    assert np.array(vs_getter.get_submat([3, 1], [0, 2])).tolist() == [[9.0, 11.0], [3.0, 5.0]]
    assert np.array(vs_getter.get_submat([4], [1, 0, 2])).tolist() == [[13.0, 12.0, 14.0]]
