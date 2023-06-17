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
from abc import abstractmethod
from ctypes import (
    c_bool,
    c_uint32,
    c_uint64,
    c_char_p,
    c_void_p,
)
from pecos.core import clib


class MmapHashmap(object):
    """
    Python wrapper of Memory-mappable Hashmap
    """

    def __init__(self, map_type):
        if map_type not in clib.mmap_map_fn_dict:
            raise NotImplementedError(f"map_type={map_type} is not implemented.")

        self.map_type = map_type
        self.map = None
        self.mode = None
        self.map_dir = None

    def open(self, mode, map_dir):
        if mode == "w":
            map = _MmapHashmapWrite.init(self.map_type, map_dir)
        elif mode == "r" or mode == "r_lazy":
            lazy_load = True if mode == "r_lazy" else False
            map = _MmapHashmapReadOnly.init(self.map_type, map_dir, lazy_load)
        else:
            raise NotImplementedError(f"{mode} not implemented.")

        self.map = map
        self.mode = mode
        self.map_dir = map_dir

    def close(self):
        if self.mode == "w":
            self.map.save()
        self.map.destruct()


class _MmapHashmapBase(object):
    def __init__(self, map_ptr, fn_dict):
        self.map_ptr = map_ptr
        self.fn_dict = fn_dict

    def size(self):
        return self.fn_dict["size"](self.map_ptr)

    def destruct(self):
        self.fn_dict["destruct"](self.map_ptr)


class _MmapHashmapReadOnly(_MmapHashmapBase):
    @abstractmethod
    def get(self, key, default_val):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __contains__(self, key):
        pass

    @classmethod
    def init(cls, map_type, map_dir, lazy_load):
        fn_dict = clib.mmap_hashmap_init(map_type)
        map_ptr = fn_dict["load"](c_char_p(map_dir.encode("utf-8")), c_bool(lazy_load))

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
        """
        return self.fn_dict["get_w_default"](
            c_void_p(self.map_ptr),
            c_char_p(key_utf8),
            c_uint32(len(key_utf8)),
            c_uint64(default_val),
        )

    def __getitem__(self, key_utf8):
        return self.fn_dict["get"](
            c_void_p(self.map_ptr), c_char_p(key_utf8), c_uint32(len(key_utf8))
        )

    def __contains__(self, key_utf8):
        return self.fn_dict["contains"](
            c_void_p(self.map_ptr), c_char_p(key_utf8), c_uint32(len(key_utf8))
        )


class _MmapHashmapInt2IntReadOnly(_MmapHashmapReadOnly):
    def get(self, key, default_val):
        return self.fn_dict["get_w_default"](
            c_void_p(self.map_ptr), c_uint64(key), c_uint64(default_val)
        )

    def __getitem__(self, key):
        return self.fn_dict["get"](c_void_p(self.map_ptr), c_uint64(key))

    def __contains__(self, key):
        return self.fn_dict["contains"](c_void_p(self.map_ptr), c_uint64(key))


class _MmapHashmapWrite(_MmapHashmapBase):
    def __init__(self, map_ptr, fn_dict, map_dir):
        super().__init__(map_ptr, fn_dict)
        self.map_dir = map_dir

    @abstractmethod
    def insert(self, key, val):
        pass

    def save(self):
        import pathlib

        pathlib.Path(self.map_dir).mkdir(parents=True, exist_ok=True)
        self.fn_dict["save"](self.map_ptr, c_char_p(self.map_dir.encode("utf-8")))

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
        self.fn_dict["insert"](
            c_void_p(self.map_ptr), c_char_p(key_utf8), c_uint32(len(key_utf8)), c_uint64(val)
        )


class _MmapHashmapInt2IntWrite(_MmapHashmapWrite):
    def insert(self, key, val):
        self.fn_dict["insert"](c_void_p(self.map_ptr), c_uint64(key), c_uint64(val))
