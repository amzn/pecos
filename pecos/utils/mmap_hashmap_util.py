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
from typing import Optional, Tuple
from ctypes import c_char_p, c_uint32, c_uint64, POINTER
import numpy as np
import os

LOGGER = logging.getLogger(__name__)


class MmapHashmap(object):
    """
    Python wrapper of Memory-mappable Hashmap, which is similar to Python Dict,
    but provides memory-map save/load functionalities, so that user could write and dump to disk,
    and later load read-only with memory-map techniques which allows out-of-core
    access for large data that cannot fit into memory.

    However, Memory-mappable Hashmap is not identical to Python Dict.
    The major differences are:
        * Only read-only and write-only modes are provided.
        * In write-only mode, deleting key is not allowed. Can only do following operations:
            * Insert key
            * Overwrite value for an existing key
    """

    def __init__(self, map_type: str):
        if map_type not in clib.mmap_map_fn_dict:
            raise NotImplementedError(f"map_type={map_type} is not implemented.")

        self.map_type = map_type
        self.map = None
        self.mode: Optional[str] = None
        self.map_dir: Optional[str] = None

    def open(self, mode: str, map_dir: str):
        """
        Open hashmap at given directory for read-only or write-only.
        For write-only, the hashmap exists in RAM until dumps to given directory when closing.

        args:
            mode: Open mode, in "w", "r", "r_lazy"(not pre-load).
            map_dir: Directory to load from/save to.
        """
        if mode == "w":
            map = _MmapHashmapWrite.init(self.map_type, map_dir)
            LOGGER.info(f"Opened hashmap for writing. Will save to {map_dir} upon closing.")
        elif mode == "r" or mode == "r_lazy":
            lazy_load = True if mode == "r_lazy" else False
            map = _MmapHashmapReadOnly.init(self.map_type, map_dir, lazy_load)
            LOGGER.info(
                f"Opened hashmap for read-only. Will {'NOT' if lazy_load else ''} pre-load."
            )
        else:
            raise NotImplementedError(f"{mode} not implemented.")

        self.map = map
        self.mode = mode
        self.map_dir = map_dir

    def close(self):
        """
        Close and destruct hashmap.
        For write-only, dumps to given directory.
        """
        if self.mode == "w":
            self.map.save()
            LOGGER.info(f"Saved hashmap to {self.map_dir} upon closing.")
        self.map.destruct()
        LOGGER.info("Destructed hashmap upon closing.")
        self.map = None
        self.mode = None
        self.map_dir = None

    def __del__(self):
        """
        Destructor to call close() if not called previously
        """
        if self.map is not None:
            self.close()


class MmapHashmapBatchGetter(object):
    """
    Batch getter for MmapHashmap opened for readonly.
    """

    def __init__(self, mmap_r, max_batch_size: int, threads: int = 1):
        if not isinstance(mmap_r, _MmapHashmapReadOnly):
            raise ValueError(f"Should get from readonly MmapHashmap, got {type(mmap_r)}")
        if max_batch_size <= 0:
            raise ValueError(f"Max batch size should >0, got {max_batch_size}")
        if threads <= 0 and threads != -1:
            raise ValueError(f"Number of threads should >0 or =-1, got {threads}")

        self.mmap_r: Optional[_MmapHashmapReadOnly] = mmap_r
        self.max_batch_size = max_batch_size
        self.key_prealloc = mmap_r.get_keyalloc(max_batch_size)

        # `os.cpu_count()` is not equivalent to the number of CPUs the current process can use.
        # The number of usable CPUs can be obtained with len(os.sched_getaffinity(0))
        n_usable_cpu = len(os.sched_getaffinity(0))
        self.threads_c_uint32 = c_uint32(
            min(n_usable_cpu, n_usable_cpu if threads == -1 else threads)
        )

        # Pre-allocated space for returns
        self.vals = np.zeros(max_batch_size, dtype=np.uint64)

    def get(self, keys, default_val):
        """
        Batch get multiple keys' values. For non-exist keys, `default_val` is returned.

        NOTE:
            1) Make sure keys given is compatible with the `MmapHashmap` `batch_get` type.
                i) str2int: List of UTF8 encoded strings
                ii) int2int: 1D numpy array of int64
            2) The return is a reused buffer, use or copy the data once you get it. It is not guaranteed to last.
        """

        if len(keys) > self.max_batch_size:
            self.max_batch_size = max(len(keys), 2 * self.max_batch_size)
            self.key_prealloc = self.mmap_r.get_keyalloc(self.max_batch_size)
            self.vals = np.zeros(self.max_batch_size, dtype=np.uint64)
            LOGGER.info(f"Increased the max batch size to {self.max_batch_size}")

        self.mmap_r.batch_get(
            len(keys),
            self.key_prealloc.get_key_prealloc(keys),
            default_val,
            self.vals,
            self.threads_c_uint32,
        )
        return memoryview(self.vals)[: len(keys)]


class _MmapHashmapBase(object):
    """Base class for methods shared by all modes"""

    def __init__(self, map_ptr, fn_dict):
        self.map_ptr = map_ptr
        self.fn_dict = fn_dict

    def size(self):
        return self.fn_dict["size"](self.map_ptr)

    def destruct(self):
        self.fn_dict["destruct"](self.map_ptr)


class _MmapHashmapReadOnly(_MmapHashmapBase):
    """Base class for methods shared by all read modes"""

    @abstractmethod
    def get(self, key, default_val):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __contains__(self, key):
        pass

    @abstractmethod
    def batch_get(self, n_keys, keys, default_val, vals, threads_c_uint32):
        pass

    @classmethod
    @abstractmethod
    def get_keyalloc(cls, max_batch_size):
        pass

    @classmethod
    def init(cls, map_type, map_dir, lazy_load):
        fn_dict = clib.mmap_hashmap_init(map_type)
        map_ptr = fn_dict["load"](map_dir.encode("utf-8"), lazy_load)

        if map_type == "str2int":
            return _MmapHashmapStr2IntReadOnly(map_ptr, fn_dict)
        elif map_type == "int2int":
            return _MmapHashmapInt2IntReadOnly(map_ptr, fn_dict)
        else:
            raise NotImplementedError(f"{map_type}")


class _MmapHashmapStr2IntReadOnly(_MmapHashmapReadOnly):
    def get(self, key_utf8, default_val):
        """
        Args:
            key_utf8: UTF8 encoded bytes string key
            default_val: Default value for key not found
        """
        return self.fn_dict["get_w_default"](
            self.map_ptr,
            key_utf8,
            len(key_utf8),
            default_val,
        )

    def __getitem__(self, key_utf8):
        return self.fn_dict["get"](self.map_ptr, key_utf8, len(key_utf8))

    def __contains__(self, key_utf8):
        return self.fn_dict["contains"](self.map_ptr, key_utf8, len(key_utf8))

    def batch_get(
        self, n_keys: int, keys_utf8: Tuple, default_val: int, vals, threads_c_uint32: c_uint32
    ):
        """
        Batch get values for UTF8 encoded bytes string keys.
        Return values are stored in vals.

        How to make inputs from UTF8 encoded bytes string keys List `keys_utf8`:
            > keys_ptr = (c_char_p * n_keys)()
            > keys_ptr[:] = keys_utf8
            > keys_lens = np.array([len(k) for k in keys_utf8], dtype=np.uint32)

        Args:
            n_keys: int. Number of keys to get.
            keys_utf8: Tuple of (keys_ptr, keys_lens)
                keys_ptr: List of UTF8 encoded bytes string keys' pointers
                keys_lens: 1D Int32 Numpy array of string keys' lengths
            default_val: Default value for key not found
            vals: 1D Int64 Numpy array to return results
            threads_c_uint32: Number of threads to use.
        """
        keys_ptr, keys_lens = keys_utf8
        self.fn_dict["batch_get_w_default"](
            self.map_ptr,
            n_keys,
            keys_ptr,
            keys_lens.ctypes.data_as(POINTER(c_uint32)),
            default_val,
            vals.ctypes.data_as(POINTER(c_uint64)),
            threads_c_uint32,
        )
        return vals

    @classmethod
    def get_keyalloc(cls, max_batch_size):
        return _Str2IntBatchGetterKeyPreAlloc(max_batch_size)


class _Str2IntBatchGetterKeyPreAlloc(object):
    """
    Key pre-allocate for Str2Int MmapHashmap.
    """

    def __init__(self, max_batch_size: int):
        self.keys_ptr = (c_char_p * max_batch_size)()
        self.keys_lens = np.zeros(max_batch_size, dtype=np.uint32)

    def get_key_prealloc(self, keys_utf8):
        self.keys_ptr[: len(keys_utf8)] = keys_utf8
        self.keys_lens.flat[: len(keys_utf8)] = [len(k) for k in keys_utf8]

        return (self.keys_ptr, self.keys_lens)


class _MmapHashmapInt2IntReadOnly(_MmapHashmapReadOnly):
    def get(self, key, default_val):
        return self.fn_dict["get_w_default"](self.map_ptr, key, default_val)

    def __getitem__(self, key):
        return self.fn_dict["get"](self.map_ptr, key)

    def __contains__(self, key):
        return self.fn_dict["contains"](self.map_ptr, key)

    def batch_get(self, n_keys: int, keys, default_val: int, vals, threads_c_uint32: c_uint32):
        """
        Batch get values for Int64 keys.
        Return values are stored in vals.

        Args:
            n_keys: int. Number of keys to get.
            keys: 1D Int64 Numpy array
            default_val: Default value for key not found
            vals: 1D Int64 Numpy array to return results
            threads_c_uint32: Number of threads to use.
        """
        self.fn_dict["batch_get_w_default"](
            self.map_ptr,
            n_keys,
            keys.ctypes.data_as(POINTER(c_uint64)),
            default_val,
            vals.ctypes.data_as(POINTER(c_uint64)),
            threads_c_uint32,
        )
        return vals

    @classmethod
    def get_keyalloc(cls, max_batch_size):
        return _Int2IntBatchGetterKeyPreAlloc(max_batch_size)


class _Int2IntBatchGetterKeyPreAlloc(object):
    """
    Dummy key pre-allocate for Int2Int MmapHashmap.
    """

    def __init__(self, max_batch_size: int):
        pass

    def get_key_prealloc(self, keys):
        return keys


class _MmapHashmapWrite(_MmapHashmapBase):
    """Base class for methods shared by all write modes"""

    def __init__(self, map_ptr, fn_dict, map_dir):
        super().__init__(map_ptr, fn_dict)
        self.map_dir = map_dir

    @abstractmethod
    def insert(self, key, val):
        pass

    def save(self):
        import pathlib

        pathlib.Path(self.map_dir).mkdir(parents=True, exist_ok=True)
        self.fn_dict["save"](self.map_ptr, (self.map_dir.encode("utf-8")))

    @classmethod
    def init(cls, map_type, map_dir):
        fn_dict = clib.mmap_hashmap_init(map_type)
        map_ptr = fn_dict["new"]()

        if map_type == "str2int":
            return _MmapHashmapStr2IntWrite(map_ptr, fn_dict, map_dir)
        elif map_type == "int2int":
            return _MmapHashmapInt2IntWrite(map_ptr, fn_dict, map_dir)
        else:
            raise NotImplementedError(f"{map_type}")


class _MmapHashmapStr2IntWrite(_MmapHashmapWrite):
    def insert(self, key_utf8, val):
        """
        Args:
            key_utf8 (bytes): UTF8 encoded bytes string key
            val (int): Integer value
        """
        self.fn_dict["insert"](self.map_ptr, key_utf8, len(key_utf8), val)


class _MmapHashmapInt2IntWrite(_MmapHashmapWrite):
    def insert(self, key, val):
        self.fn_dict["insert"](self.map_ptr, key, val)
