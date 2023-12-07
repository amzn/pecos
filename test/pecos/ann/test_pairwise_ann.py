#  Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import random  # noqa
import numpy as np  # noqa
import scipy.sparse as smat  # noqa
from pytest import approx  # noqa


def test_importable():
    from pecos.ann.pairwise import PairwiseANN  # noqa: F401


def test_save_and_load(tmpdir):
    from pecos.ann.pairwise import PairwiseANN  # noqa: F401

    # constant
    N = 10000
    D = 64
    max_bsz, only_topk = 250, 10
    X_drm = np.random.randn(N, D).astype(np.float32)
    Y_csc = smat.eye(N, dtype=np.float32, format="csc")
    train_params = PairwiseANN.TrainParams(metric_type="ip")
    pred_params = PairwiseANN.PredParams(batch_size=max_bsz, only_topk=only_topk)
    label_keys = np.arange(max_bsz).astype(np.uint32)

    # train, predict and save
    model = PairwiseANN.train(X_drm, Y_csc, train_params=train_params)
    searchers = model.searchers_create(pred_params=pred_params, num_searcher=1)
    It, Mt, Dt, Vt = model.predict(
        X_drm[:max_bsz],
        label_keys,
        searchers,
        is_same_input=False,
    )
    model_folder = tmpdir.join("pairwise_ann_dir")
    model.save(model_folder)
    del model, searchers

    # load w/ mmap and predict again
    model = PairwiseANN.load(model_folder, lazy_load=True)
    searchers = model.searchers_create(pred_params=pred_params, num_searcher=2)
    Ip, Mp, Dp, Vp = model.predict(
        X_drm[:max_bsz],
        label_keys,
        searchers,
        is_same_input=False,
    )
    assert Ip == approx(It, abs=0.0), f"pred faield: Ip != It"
    assert Mp == approx(Mt, abs=0.0), f"pred faield: Mp != Mt"
    assert Dp == approx(Dt, abs=0.0), f"pred faield: Dp != Dt"
    assert Vp == approx(Vt, abs=0.0), f"pred faield: Vp != Vt"
    del model, searchers


def test_consistency_between_drm_and_csr():
    from pecos.ann.pairwise import PairwiseANN  # noqa: F401

    # constant
    N = D = 128
    max_bsz, only_topk = 3, 2
    X_drm = np.random.randn(N, D).astype(np.float32)
    X_spr = smat.csr_matrix(X_drm)
    Y_csc = smat.eye(N, dtype=np.float32, format="csc")
    train_params = PairwiseANN.TrainParams(metric_type="ip")
    pred_params = PairwiseANN.PredParams(batch_size=max_bsz, only_topk=only_topk)
    label_keys = np.arange(max_bsz).astype(np.uint32)

    # case 1: dense input feat
    model_dns = PairwiseANN.train(X_drm, Y_csc, train_params=train_params)
    searchers = model_dns.searchers_create(pred_params=pred_params, num_searcher=1)
    I_dns, M_dns, D_dns, V_dns = model_dns.predict(
        X_drm[:max_bsz],
        label_keys,
        searchers,
        is_same_input=False,
    )

    # test outcome accuracy
    assert I_dns == approx(
        np.array([[0, 0], [1, 0], [2, 0]]),
        abs=0.0,
    ), f"pred failed: I_dns != I_true"
    assert M_dns == approx(
        np.array([[1, 0], [1, 0], [1, 0]]),
        abs=0.0,
    ), f"pred failed: M_dns != M_true"
    assert V_dns == approx(
        np.array([[1, 0], [1, 0], [1, 0]]),
        abs=0.0,
    ), f"pred failed: V_dns != V_true"

    # case 2: sparse input feat
    model_spr = PairwiseANN.train(X_spr, Y_csc, train_params=train_params)
    searchers = model_spr.searchers_create(pred_params=pred_params, num_searcher=2)
    I_spr, M_spr, D_spr, V_spr = model_spr.predict(
        X_spr[:max_bsz],
        label_keys,
        searchers,
        is_same_input=False,
    )

    # test outcome consistency between sparse/dense
    assert I_dns == approx(I_spr, abs=0.0), f"pred failed: I_dns != I_spr"
    assert M_dns == approx(M_spr, abs=0.0), f"pred failed: M_dns != M_spr"
    assert D_dns == approx(D_spr, abs=1e-4), f"pred failed: D_dns != D_spr"
    assert V_dns == approx(V_spr, abs=0.0), f"pred failed: V_dns != V_spr"
    del model_dns, model_spr, searchers


def test_predict_with_same_input():
    from pecos.ann.pairwise import PairwiseANN  # noqa: F401

    # constant
    X_drm = np.array(
        [
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0],
            [5, 0],
        ]
    ).astype(np.float32)
    Y_csc = smat.csr_matrix(
        np.array(
            [
                [1.1, 0.0, 0.0, 0.0],
                [2.1, 2.2, 0.0, 0.0],
                [0.0, 3.2, 3.3, 0.0],
                [0.0, 0.0, 4.3, 4.4],
                [0.0, 0.0, 0.0, 5.4],
            ]
        ).astype(np.float32)
    )
    max_bsz, only_topk = 4, 3
    label_keys = np.arange(max_bsz).astype(np.uint32)
    train_params = PairwiseANN.TrainParams(metric_type="ip")
    pred_params = PairwiseANN.PredParams(batch_size=max_bsz, only_topk=only_topk)
    label_keys = np.arange(max_bsz).astype(np.uint32)

    # train & predict
    model = PairwiseANN.train(X_drm, Y_csc, train_params=train_params)
    searchers = model.searchers_create(pred_params=pred_params, num_searcher=1)
    Ip, Mp, Dp, Vp = model.predict(
        X_drm[:max_bsz],
        label_keys,
        searchers,
        is_same_input=False,
    )
    assert Ip == approx(
        np.array([[1, 0, 0], [2, 1, 0], [3, 2, 0], [4, 3, 0]]),
        abs=0.0,
    ), f"pred failed: Ip != I_true"
    assert Mp == approx(
        np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]),
        abs=0.0,
    ), f"pred failed: Mp != M_true"
    assert Dp == approx(
        np.array([[-1, 0, 0], [-5, -3, 0], [-11, -8, 0], [-19, -15, 0]]),
        abs=0.0,
    ), f"pred failed: Dp != D_true"
    assert Vp == approx(
        np.array([[2.1, 1.1, 0], [3.2, 2.2, 0], [4.3, 3.3, 0], [5.4, 4.4, 0]]),
        abs=1e-6,
    ), f"pred failed: Vp != V_true"


def test_predict_with_multiple_calls():
    from pecos.ann.pairwise import PairwiseANN

    # constant
    N = L = 8192
    D = 64
    X_drm = np.random.randn(N, D).astype(np.float32)
    Y_csc = smat.random(N, L, density=3e-4, format="csc", dtype=np.float32)
    max_bsz, only_topk = 10, 3
    label_keys = np.arange(max_bsz).astype(np.uint32)
    train_params = PairwiseANN.TrainParams(metric_type="ip")
    pred_params = PairwiseANN.PredParams(batch_size=max_bsz, only_topk=only_topk)
    label_keys = np.arange(max_bsz).astype(np.uint32)

    # case 1: is_same_input = False
    X_tst = np.random.randn(2, D).astype(np.float32)
    X_tst_dup = np.repeat(X_tst, 5, axis=0)  # [x1,x2] => [x1,...,x1,x2,...,x2]
    model = PairwiseANN.train(X_drm, Y_csc, train_params=train_params)
    searchers = model.searchers_create(pred_params=pred_params, num_searcher=1)
    Ip, Mp, Dp, Vp = model.predict(
        X_tst_dup,
        label_keys,
        searchers,
        is_same_input=False,
    )

    # case 2: 1st half and is_same_input = True
    Ip1, Mp1, Dp1, Vp1 = model.predict(
        X_tst[0, :].reshape(1, -1),
        label_keys[:5],
        searchers,
        is_same_input=True,
    )
    assert Ip1 == approx(Ip[:5, :], abs=0.0), f"pred failed: Ip1 != Ip"
    assert Mp1 == approx(Mp[:5, :], abs=0.0), f"pred failed: Mp1 != Mp"
    assert Dp1 == approx(Dp[:5, :], abs=0.0), f"pred failed: Dp1 != Dp"
    assert Vp1 == approx(Vp[:5, :], abs=0.0), f"pred failed: Vp1 != Vp"

    # case 3: 2nd half and is_same_input = True
    Ip2, Mp2, Dp2, Vp2 = model.predict(
        X_tst[1, :].reshape(1, -1),
        label_keys[5:],
        searchers,
        is_same_input=True,
    )
    assert Ip2 == approx(Ip[5:, :], abs=0.0), f"pred failed: Ip2 != Ip"
    assert Mp2 == approx(Mp[5:, :], abs=0.0), f"pred failed: Mp2 != Mp"
    assert Dp2 == approx(Dp[5:, :], abs=0.0), f"pred failed: Dp2 != Dp"
    assert Vp2 == approx(Vp[5:, :], abs=0.0), f"pred failed: Vp2 != Vp"
