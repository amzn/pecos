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


def test_str2int_mmap_hashmap(tmpdir):
    from pecos.utils.mmap_hashmap_util import MmapHashmap, MmapHashmapBatchGetter

    map_dir = tmpdir.join("str2int_mmap").realpath().strpath
    kv_dict = {"aaaa".encode("utf-8"): 2, "bb".encode("utf-8"): 3}

    # Write-only Mode
    w_map = MmapHashmap("str2int")
    w_map.open("w", map_dir)
    # Insert
    w_map.map.insert("aaaa".encode("utf-8"), 1)  # Test for overwrite later
    for k, v in kv_dict.items():
        w_map.map.insert(k, v)
    # Size
    assert w_map.map.size() == len(kv_dict)
    w_map.close()

    # Read-only Mode
    r_map = MmapHashmap("str2int")
    r_map.open("r", map_dir)
    # Get
    for k, v in kv_dict.items():
        assert r_map.map[k] == v
    # Get with default
    for k, v in kv_dict.items():
        assert r_map.map.get(k, 10) == v
    assert r_map.map.get("ccccc".encode("utf-8"), 10) == 10
    # Contains
    for k, _ in kv_dict.items():
        assert k in r_map.map
    assert not ("ccccc".encode("utf-8") in r_map.map)
    # Size
    assert r_map.map.size() == len(kv_dict)

    # Batch get with default
    max_batch_size = 5
    # max_batch_size > num of key
    r_map_batch_getter = MmapHashmapBatchGetter(r_map.map, max_batch_size)
    ks = list(kv_dict.keys()) + ["ccccc".encode("utf-8")]  # Non-exist key
    vs = list(kv_dict.values()) + [10]
    assert r_map_batch_getter.get(ks, 10).tolist() == vs
    # max_batch_size = num of key
    ks = list(kv_dict.keys()) + ["ccccc".encode("utf-8")] * (
        max_batch_size - len(kv_dict)
    )  # Non-exist key
    vs = list(kv_dict.values()) + [10] * (max_batch_size - len(kv_dict))
    assert r_map_batch_getter.get(ks, 10).tolist() == vs
    # max_batch_size = num of key * 3
    ks = list(kv_dict.keys()) + ["ccccc".encode("utf-8")] * (
        3 * max_batch_size - len(kv_dict)
    )  # Non-exist key
    vs = list(kv_dict.values()) + [10] * (3 * max_batch_size - len(kv_dict))
    assert r_map_batch_getter.get(ks, 10).tolist() == vs


def test_int2int_mmap_hashmap(tmpdir):
    from pecos.utils.mmap_hashmap_util import MmapHashmap, MmapHashmapBatchGetter
    import numpy as np

    map_dir = tmpdir.join("int2int_mmap").realpath().strpath
    kv_dict = {10: 2, 20: 3}

    # Write-only Mode
    w_map = MmapHashmap("int2int")
    w_map.open("w", map_dir)
    # Insert
    w_map.map.insert(10, 1)  # Test for overwrite later
    for k, v in kv_dict.items():
        w_map.map.insert(k, v)
    # Size
    assert w_map.map.size() == len(kv_dict)
    w_map.close()

    # Read-only Mode
    r_map = MmapHashmap("int2int")
    r_map.open("r", map_dir)
    # Get
    for k, v in kv_dict.items():
        assert r_map.map[k] == v
    # Get with default
    for k, v in kv_dict.items():
        assert r_map.map.get(k, 10) == v
    assert r_map.map.get(1000, 10) == 10
    # Contains
    for k, _ in kv_dict.items():
        assert k in r_map.map
    assert not (1000 in r_map.map)
    # Size
    assert r_map.map.size() == len(kv_dict)

    # Batch get with default
    max_batch_size = 5
    # max_batch_size > num of key
    r_map_batch_getter = MmapHashmapBatchGetter(r_map.map, max_batch_size)
    ks = list(kv_dict.keys()) + [1000]  # Non-exist key
    vs = list(kv_dict.values()) + [10]
    assert r_map_batch_getter.get(np.array(ks, dtype=np.int64), 10).tolist() == vs
    # max_batch_size = num of key
    ks = list(kv_dict.keys()) + [1000] * (max_batch_size - len(kv_dict))  # Non-exist key
    vs = list(kv_dict.values()) + [10] * (max_batch_size - len(kv_dict))
    assert r_map_batch_getter.get(np.array(ks, dtype=np.int64), 10).tolist() == vs
    # max_batch_size = num of key * 3
    ks = list(kv_dict.keys()) + [1000] * (3 * max_batch_size - len(kv_dict))  # Non-exist key
    vs = list(kv_dict.values()) + [10] * (3 * max_batch_size - len(kv_dict))
    assert r_map_batch_getter.get(np.array(ks, dtype=np.int64), 10).tolist() == vs
