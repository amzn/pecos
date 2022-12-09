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

from pytest import approx


def test_importable():
    import pecos.utils.cluster_util  # noqa: F401
    from pecos.utils.cluster_util import ClusterChain  # noqa: F401


def test_cluster_chain(tmpdir):
    import scipy.sparse as smat
    import numpy as np
    from pecos.utils.cluster_util import ClusterChain

    C0 = smat.csc_matrix([[1], [1]], dtype=np.float32)
    C1 = smat.csc_matrix([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    C2 = smat.csc_matrix(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    chain_orig = ClusterChain([C0, C1, C2])

    # test save & load
    save_path = tmpdir.join("chain")
    chain_orig.save(save_path)
    chain_loaded = ClusterChain.load(save_path)
    assert chain_orig == chain_loaded

    # test chain construction
    chain_reconstructed = ClusterChain.from_partial_chain(chain_orig[-1], nr_splits=2)
    assert chain_orig == chain_reconstructed

    # test matching chain
    Y = smat.csr_matrix([[1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]], dtype=np.float32)
    M2 = smat.csr_matrix([[0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    matching_chain = chain_orig.generate_matching_chain({0: Y, 1: M2, 2: None})
    M2_res = np.array([[1, 1, 1, 0], [0, 1, 0, 1]])
    M1_res = np.array([[2, 1], [1, 1]])
    M0_res = np.array([[3], [2]])
    for pred, res in zip(matching_chain, [M0_res, M1_res, M2_res]):
        assert pred.toarray() == approx(res)

    # test relevance chain
    R = smat.csr_matrix(
        [[2.0, 0, 0, 0, 0, 0.5, 0, 0], [0, 0, 0.1, 0, 0, 0, 0, 0]], dtype=np.float32
    )
    R2 = smat.csr_matrix([[0.2, 0.2], [10.0, 0]], dtype=np.float32)

    R0_res = np.array([[0.5, 0.5], [1.0, 0]], dtype=np.float32)
    R1_res = np.array([[0.8, 0.0, 0.2, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    R2_res = np.array([[0.8, 0, 0, 0, 0, 0.2, 0, 0], [0, 0, 1.0, 0, 0, 0, 0, 0]], dtype=np.float32)
    relevance_chain = chain_orig.generate_relevance_chain({0: R, 2: R2, 3: None}, norm_type="l1")
    for pred, res in zip(relevance_chain, [R0_res, R1_res, R2_res]):
        assert pred.toarray() == approx(res)
    relevance_chain = chain_orig.generate_relevance_chain({0: R}, norm_type="l1", induce=False)
    for pred, res in zip(relevance_chain, [None, None, R2_res]):
        if res is None:
            assert pred is None
        else:
            assert pred.toarray() == approx(res)
