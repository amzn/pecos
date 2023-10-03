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
import logging
from abc import abstractmethod
from pecos.core import clib
from typing import Tuple, Optional
from ctypes import c_char_p, c_uint32, c_uint64, c_float, POINTER
import numpy as np
import os

LOGGER = logging.getLogger(__name__)


class MmapValStore(object):
    """
    Python wrapper of memory-mapped 2D matrix value store.
    """

    def __init__(self, store_type: str):
        if store_type not in clib.mmap_valstore_fn_dict:
            raise NotImplementedError(f"store_type={store_type} is not implemented.")

        self.store_type = store_type
        self.store = None
        self.mode: Optional[str] = None
        self.store_dir: Optional[str] = None

    def open(self, mode: str, store_dir: str):
        """
        Open value store at given directory for read-only or write-only.
        For write-only, the value store exists in RAM until dumps to given directory when closing.

        args:
            mode: Open mode, in "w", "r", "r_lazy"(not pre-load).
            store_dir: Directory to load from/save to.
        """
        if mode == "w":
            store = _MmapValStoreWrite.init(self.store_type, store_dir)
            LOGGER.info(f"Opened value store for writing. Will save to {store_dir} upon closing.")
        elif mode == "r" or mode == "r_lazy":
            lazy_load = True if mode == "r_lazy" else False
            store = _MmapValStoreReadOnly.init(self.store_type, store_dir, lazy_load)
            LOGGER.info(
                f"Opened value store for read-only. Will {'NOT' if lazy_load else ''} pre-load."
            )
        else:
            raise NotImplementedError(f"{mode} not implemented.")

        self.store = store
        self.mode = mode
        self.store_dir = store_dir

    def close(self):
        """
        Close and destruct value store.
        For write-only, dumps to given directory.
        """
        if self.mode == "w":
            self.store.save()
            LOGGER.info(f"Saved value store to {self.store_dir} upon closing.")
        self.store.destruct()
        LOGGER.info("Destructed value store upon closing.")
        self.store = None
        self.mode = None
        self.store_dir = None

    def __del__(self):
        """
        Destructor to call close() if not called previously
        """
        if self.store is not None:
            self.close()


class MmapValStoreSubMatGetter(object):
    """
    Sub-matrix getter for MmapValStore opened for readonly.

    Args:
        max_row_size: Maximum row size
        max_col_size: Maximum column size
        trunc_val_len (Optional): Applicable for string value only. Truncated max string length.
        threads (Optional): Number of threads to use.
    """

    def __init__(
        self,
        store_r,
        max_row_size: int,
        max_col_size: int,
        trunc_val_len: int = 256,
        threads: int = 1,
    ):
        if not isinstance(store_r, _MmapValStoreReadOnly):
            raise ValueError(f"Should get from readonly MmapValStore, got {type(store_r)}")
        if max_row_size <= 0:
            raise ValueError(f"Max row size should >0, got {max_row_size}")
        if max_col_size <= 0:
            raise ValueError(f"Max col size should >0, got {max_col_size}")
        if threads <= 0 and threads != -1:
            raise ValueError(f"Number of threads should >0 or =-1, got {threads}")

        self.store_r: Optional[_MmapValStoreReadOnly] = store_r

        # `os.cpu_count()` is not equivalent to the number of CPUs the current process can use.
        # The number of usable CPUs can be obtained with len(os.sched_getaffinity(0))
        n_usable_cpu = len(os.sched_getaffinity(0))
        self.threads_c_uint32 = c_uint32(
            min(n_usable_cpu, n_usable_cpu if threads == -1 else threads)
        )

        # Pre-allocated space for sub-matrix row & col indices
        self.sub_rows = np.zeros(max_row_size, dtype=np.uint64)
        self.sub_cols = np.zeros(max_col_size, dtype=np.uint32)
        self.sub_rows_ptr = self.sub_rows.ctypes.data_as(POINTER(c_uint64))
        self.sub_cols_ptr = self.sub_cols.ctypes.data_as(POINTER(c_uint32))

        # Pre-allocated space for return values sub-matrix
        self.val_prealloc = store_r.get_val_alloc(max_row_size, max_col_size, trunc_val_len)

    def get_submat(self, rows, cols):
        """
        Get sub-matrix of the value store with given indices of rows and columns.

        NOTE:
            1) `rows` & `cols` are list-like objects,
                e.g. List, np array, memoryview returned from MmapHashmap batch_get
            2) The return is a reused buffer, use or copy the data once you get it. It is not guaranteed to last.
        """
        n_rows = len(rows)
        n_cols = len(cols)
        self.sub_rows.flat[:n_rows] = rows
        self.sub_cols.flat[:n_cols] = cols
        self.store_r.get_submatrix(
            n_rows,
            n_cols,
            self.sub_rows_ptr,
            self.sub_cols_ptr,
            self.val_prealloc.ret_vals,
            self.threads_c_uint32,
        )
        return self.val_prealloc.get_ret_memoryview(n_rows, n_cols)


class _MmapValStoreBase(object):
    """Base class for methods shared by all modes"""

    def __init__(self, store_ptr, fn_dict):
        self.store_ptr = store_ptr
        self.fn_dict = fn_dict

    def n_row(self):
        return self.fn_dict["n_row"](self.store_ptr)

    def n_col(self):
        return self.fn_dict["n_col"](self.store_ptr)

    def destruct(self):
        self.fn_dict["destruct"](self.store_ptr)


class _MmapValStoreReadOnly(_MmapValStoreBase):
    """Base class for methods shared by all read modes"""

    @abstractmethod
    def get_submatrix(self, n_rows, n_cols, rows_ptr, cols_ptr, ret_vals: Tuple, threads_c_uint32):
        pass

    @classmethod
    @abstractmethod
    def get_val_alloc(cls, max_row_size: int, max_col_size: int, trunc_val_len: int = 256):
        """
        Get reusable return value pre-allocations.
        """
        pass

    @classmethod
    def init(cls, store_type, store_dir, lazy_load):
        fn_dict = clib.mmap_valstore_init(store_type)
        store_ptr = fn_dict["load"](store_dir.encode("utf-8"), lazy_load)

        if store_type == "num_f32":
            return _MmapValStoreNumF32ReadOnly(store_ptr, fn_dict)
        elif store_type == "str":
            return _MmapValStoreStrReadOnly(store_ptr, fn_dict)
        else:
            raise NotImplementedError(f"{store_type}")


class _MmapValStoreNumF32ReadOnly(_MmapValStoreReadOnly):
    """
    Numerical float32 value store read only implementation.
    """

    def get_submatrix(self, n_rows, n_cols, rows_ptr, cols_ptr, ret_vals, threads_c_uint32):
        self.fn_dict["get_submatrix"](
            self.store_ptr, n_rows, n_cols, rows_ptr, cols_ptr, ret_vals, threads_c_uint32
        )

    @classmethod
    def get_val_alloc(cls, max_row_size: int, max_col_size: int, trunc_val_len: int = 0):
        return _NumF32SubMatGetterValPreAlloc(max_row_size, max_col_size)


class _NumF32SubMatGetterValPreAlloc(object):
    """
    Sub-matrix return value pre-allocate for float32 Numerical MmapValStore.
    """

    def __init__(self, max_row_size: int, max_col_size: int):
        self.vals = np.zeros(max_row_size * max_col_size, dtype=np.float32)
        self.vals_ptr = self.vals.ctypes.data_as(POINTER(c_float))

        self.ret_vals = self.vals_ptr

    def get_ret_memoryview(self, n_rows, n_cols):
        """
        Reshape return into desired shape (row-major), so elements could be retrieved by indices:
            ret[i, j], 0<=i<n_rows, 0<=j<n_cols

        Return can also be assigned to row-major Numpy array of same shape:
            arr = np.zeros((n_rows, n_cols), dtype=np.float32)
            arr.flat[:] = ret
        This also works:
            arr = np.array(ret)

        Casting/Reshaping a memoryview does not copy data.
        See: https://docs.python.org/3/library/stdtypes.html#memoryview.cast
        """
        # Casting to bytes first then cast to float32 with desired shape.
        # 'f' = float32
        # For types, see: https://docs.python.org/3/library/struct.html#format-characters
        return memoryview(self.vals)[: n_rows * n_cols].cast("c").cast("f", shape=[n_rows, n_cols])


class _MmapValStoreStrReadOnly(_MmapValStoreReadOnly):
    """
    String value store read only implementation.
    """

    def get_submatrix(self, n_rows, n_cols, rows_ptr, cols_ptr, ret_vals, threads_c_uint32):
        trunc_val_len, ret_ptr, ret_lens_ptr = ret_vals
        self.fn_dict["get_submatrix"](
            self.store_ptr,
            n_rows,
            n_cols,
            rows_ptr,
            cols_ptr,
            trunc_val_len,
            ret_ptr,
            ret_lens_ptr,
            threads_c_uint32,
        )

    @classmethod
    def get_val_alloc(cls, max_row_size: int, max_col_size: int, trunc_val_len: int = 256):
        return _StrSubMatGetterValPreAlloc(max_row_size, max_col_size, trunc_val_len)


class _StrSubMatGetterValPreAlloc(object):
    """
    Sub-matrix return value pre-allocate for String MmapValStore.
    """

    def __init__(self, max_row_size: int, max_col_size: int, trunc_val_len: int):
        self.vals = np.zeros(max_row_size * max_col_size * trunc_val_len, dtype=np.dtype("b"))
        self.vals_ptr = self.vals.ctypes.data_as(c_char_p)

        self.vals_lens = np.zeros(max_row_size * max_col_size, dtype=np.uint32)
        self.vals_lens_ptr = self.vals_lens.ctypes.data_as(POINTER(c_uint32))

        self.trunc_val_len = trunc_val_len
        self.ret_vals = (c_uint32(trunc_val_len), self.vals_ptr, self.vals_lens_ptr)

    def get_ret_memoryview(self, n_rows, n_cols):
        """
        Reshape return into memoryview of bytes matrix
        """
        ret_sub_mat = (
            memoryview(self.vals)[: n_rows * n_cols * self.trunc_val_len]
            .cast("c")
            .cast("c", shape=[n_rows, n_cols, self.trunc_val_len])
        )
        len_sub_mat = (
            memoryview(self.vals_lens)[: n_rows * n_cols]
            .cast("c")
            .cast("I", shape=[n_rows, n_cols])
        )
        return (ret_sub_mat, len_sub_mat)


class _MmapValStoreWrite(_MmapValStoreBase):
    """Base class for methods shared by all write modes"""

    def __init__(self, store_ptr, fn_dict, store_dir):
        super().__init__(store_ptr, fn_dict)
        self.store_dir = store_dir

        # Stored references if necessary
        self.vals = None

    @abstractmethod
    def from_vals(self, vals):
        pass

    def save(self):
        import pathlib

        pathlib.Path(self.store_dir).mkdir(parents=True, exist_ok=True)
        self.fn_dict["save"](self.store_ptr, (self.store_dir.encode("utf-8")))

        # Delete stored references after save
        self.vals = None

    @classmethod
    def init(cls, store_type, store_dir):
        fn_dict = clib.mmap_valstore_init(store_type)
        store_ptr = fn_dict["new"]()

        if store_type == "num_f32":
            return _MmapValStoreNumF32Write(store_ptr, fn_dict, store_dir)
        elif store_type == "str":
            return _MmapValStoreStrWrite(store_ptr, fn_dict, store_dir)
        else:
            raise NotImplementedError(f"{store_type}")


class _MmapValStoreNumF32Write(_MmapValStoreWrite):
    def from_vals(self, vals):
        """
        Args:
            vals: 2D row-major numpy float32 array to write.
        """
        if not vals.flags["C_CONTIGUOUS"]:
            raise ValueError("Array is not row-major, cannot write.")
        n_row, n_col = vals.shape
        self.fn_dict["from_vals"](
            self.store_ptr, n_row, n_col, vals.ctypes.data_as(POINTER(c_float))
        )
        # Keep a reference to vals so it won't get recycled until this class instance is deleted
        self.vals = vals


class _MmapValStoreStrWrite(_MmapValStoreWrite):
    def from_vals(self, vals):
        """
        Args:
            vals: Tuple (n_row, n_col, str_list)
                n_row: Number of rows
                n_col: Number of columns
                str_list: List of UTF-8 encoded strings
        """
        n_row, n_col, str_list = vals
        n_total = n_row * n_col
        if len(str_list) != n_total:
            raise ValueError(f"Should get length {n_total} string list, got: {len(str_list)}")

        str_ptr = (c_char_p * n_total)()
        str_ptr[:] = str_list
        str_lens = np.array([len(s) for s in str_list], dtype=np.uint32)
        str_lens_ptr = str_lens.ctypes.data_as(POINTER(c_uint32))

        self.fn_dict["from_vals"](self.store_ptr, n_row, n_col, str_ptr, str_lens_ptr)
