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
from typing import Optional


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


class _MmapHashmapInt2IntReadOnly(_MmapHashmapReadOnly):
    def get(self, key, default_val):
        return self.fn_dict["get_w_default"](self.map_ptr, key, default_val)

    def __getitem__(self, key):
        return self.fn_dict["get"](self.map_ptr, key)

    def __contains__(self, key):
        return self.fn_dict["contains"](self.map_ptr, key)


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
