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

    # train data
    X_trn = np.array([[1, 0], [0, 0], [-1, 0]]).astype(np.float32)
    Y_csr = smat.csr_matrix(
        np.array(
            [
                [1.1, 0.0, 0.0],
                [0.0, 2.2, 2.4],
                [3.1, 3.2, 0.0],
            ]
        ).astype(np.float32)
    )
    # test data, noqa
    X_tst = X_trn
    label_keys = np.array([0, 1, 2]).astype(np.uint32)
    # train & predict
    train_params = PairwiseANN.TrainParams(metric_type="ip")
    model = PairwiseANN.train(X_trn, Y_csr, train_params=train_params)
    pred_params = PairwiseANN.PredParams(topk=2)
    searchers = model.searchers_create(max_batch_size=250, max_only_topk=10, num_searcher=1)
    It, Mt, Dt, Vt = model.predict(
        X_tst,
        label_keys,
        searchers,
        pred_params=pred_params,
        is_same_input=False,
    )
    # save model
    model_folder = tmpdir.join("hnsw_model_dir")
    model.save(model_folder)
    del model, searchers
    # load back and predict again
    model = PairwiseANN.load(model_folder)
    searchers = model.searchers_create(max_batch_size=250, max_only_topk=10, num_searcher=1)
    Ip, Mp, Dp, Vp = model.predict(
        X_tst,
        label_keys,
        searchers,
        pred_params=pred_params,
        is_same_input=False,
    )
    assert Ip == approx(It, abs=0.0), f"pred faield: Ip != It"
    assert Mp == approx(Mt, abs=0.0), f"pred faield: Mp != Mt"
    assert Dp == approx(Dt, abs=0.0), f"pred faield: Dp != Dt"
    assert Vp == approx(Vt, abs=0.0), f"pred faield: Vp != Vt"
    del model, searchers


def test_predict_with_same_input():
    from pecos.ann.pairwise import PairwiseANN

    # train data
    X_trn = np.array([[1, 0], [0, 0], [-1, 0]]).astype(np.float32)
    Y_csr = smat.csr_matrix(
        np.array(
            [
                [1.1, 0.0, 0.0],
                [0.0, 2.2, 2.4],
                [3.1, 3.2, 0.0],
            ]
        ).astype(np.float32)
    )
    # test data, noqa
    X_tst = np.array([[2, 0]]).astype(np.float32)
    label_keys = np.array([0, 1, 2]).astype(np.uint32)
    # train
    train_params = PairwiseANN.TrainParams(metric_type="ip")
    model = PairwiseANN.train(X_trn, Y_csr, train_params=train_params)
    # predict
    pred_params = PairwiseANN.PredParams(topk=2)
    searchers = model.searchers_create(max_batch_size=250, max_only_topk=10, num_searcher=1)
    Ip, Mp, Dp, Vp = model.predict(
        X_tst,
        label_keys,
        searchers,
        pred_params=pred_params,
        is_same_input=True,
    )
    # compare to expected ground truth
    It = np.array([[0, 2], [1, 2], [1, 0]]).astype(np.uint32)
    Mt = np.array([[1, 1], [1, 1], [1, 0]]).astype(np.uint32)
    Dt = np.array([[-1, 3], [1, 3], [1.0, 0.0]]).astype(np.float32)
    Vt = np.array([[1.1, 3.1], [2.2, 3.2], [2.4, 0.0]]).astype(np.float32)
    assert Ip == approx(It, abs=0.0), f"pred faield: Ip != It"
    assert Mp == approx(Mt, abs=0.0), f"pred faield: Mp != Mt"
    assert Dp == approx(Dt, abs=0.0), f"pred faield: Dp != Dt"
    assert Vp == approx(Vt, abs=0.0), f"pred faield: Vp != Vt"
    del model, searchers


def test_predict_with_multiple_calls():
    from pecos.ann.pairwise import PairwiseANN

    # train data
    X_trn = np.array([[1, 0], [0, 2], [-1, 0]]).astype(np.float32)
    Y_csr = smat.csr_matrix(
        np.array(
            [
                [1.1, 0.0, 0.0],
                [0.0, 2.2, 2.4],
                [3.1, 3.2, 0.0],
            ]
        ).astype(np.float32)
    )
    # test data, noqa
    batch_size = 3
    X_tst = X_trn
    label_keys = np.array(range(batch_size)).astype(np.uint32)
    # train
    train_params = PairwiseANN.TrainParams(metric_type="ip")
    model = PairwiseANN.train(X_trn, Y_csr, train_params=train_params)
    # batch predict
    pred_params = PairwiseANN.PredParams(topk=2)
    searchers = model.searchers_create(max_batch_size=250, max_only_topk=10, num_searcher=1)
    Ip, Mp, Dp, Vp = model.predict(
        X_tst,
        label_keys,
        searchers,
        pred_params=pred_params,
        is_same_input=False,
    )
    It = np.array([[0, 2], [1, 2], [1, 0]]).astype(np.uint32)
    Mt = np.array([[1, 1], [1, 1], [1, 0]]).astype(np.uint32)
    Dt = np.array([[0, 2], [-3, 1], [1, 0]]).astype(np.float32)
    Vt = np.array([[1.1, 3.1], [2.2, 3.2], [2.4, 0.0]]).astype(np.float32)
    assert Ip == approx(It, abs=0.0), f"pred failed: Ip != It"
    assert Mp == approx(Mt, abs=0.0), f"pred failed: Mp != Mt"
    assert Dp == approx(Dt, abs=0.0), f"pred failed: Dp != Dt"
    assert Vp == approx(Vt, abs=0.0), f"pred failed: Vp != Vt"

    # make predict on single (q,a) pair with multiple calls
    # to test if we properly reset the memory buffer
    for bidx in range(batch_size):
        Ip_b, Mp_b, Dp_b, Vp_b = model.predict(
            X_tst[bidx, :].reshape(1, -1),
            np.array([label_keys[bidx]]),
            searchers,
            pred_params=pred_params,
            is_same_input=True,
        )
        assert Ip_b == approx(It[bidx, :].reshape(1, -1), abs=0.0), f"bidx={bidx} failed: Ip != It"
        assert Mp_b == approx(Mt[bidx, :].reshape(1, -1), abs=0.0), f"bidx={bidx} failed: Mp != Mt"
        assert Dp_b == approx(Dt[bidx, :].reshape(1, -1), abs=0.0), f"bidx={bidx} failed: Dp != Dt"
        assert Vp_b == approx(Vt[bidx, :].reshape(1, -1), abs=0.0), f"bidx={bidx} failed: Vp != Vt"

    del model, searchers
