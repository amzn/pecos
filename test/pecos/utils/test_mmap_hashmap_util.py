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
    from pecos.utils.mmap_hashmap_util import MmapHashmap

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


def test_int2int_mmap_hashmap(tmpdir):
    from pecos.utils.mmap_hashmap_util import MmapHashmap

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
