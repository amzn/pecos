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
    import pecos.ann  # noqa: F401
    from pecos import ann  # noqa: F401
    from pecos.ann import HNSW  # noqa: F401


def test_hnsw_recall():
    import random
    import numpy as np
    import scipy.sparse as smat
    from pecos.ann import HNSW

    random.seed(1234)
    np.random.seed(1234)
    top_k = 10
    M, efC, efS = 32, 100, 50
    max_level, threads = 5, 16
    n_trn, n_tst = 90, 10
    metric_type = "ip"

    def run_one(X_trn, X_tst, Y_true):
        model = HNSW.train(X_trn, M, efC, max_level, metric_type, threads)
        Y_pred, _ = model.predict(X_tst, efS, top_k, ret_csr=False)
        recall = 0.0
        for qid in range(X_tst.shape[0]):
            yt = set(Y_true[qid, :].flatten().data)
            yp = set(Y_pred[qid, :].flatten().data)
            recall += len(yt.intersection(yp)) / top_k
        recall = recall / X_tst.shape[0]
        return recall

    # test drm
    data_dim = 2
    X_trn = np.random.uniform(low=0.0, high=1.0, size=(n_trn, data_dim)).astype(np.float32)
    X_tst = np.random.uniform(low=0.0, high=1.0, size=(n_tst, data_dim)).astype(np.float32)
    Y_true = -X_tst.dot(X_trn.T)
    Y_true = np.argsort(Y_true)[:, :top_k]
    recall = run_one(X_trn, X_tst, Y_true)
    assert recall == approx(1.0, abs=1e-2), f"hnsw with data_type=drm failed, recall={recall}"

    # test csr
    data_dim = 50
    X_trn = smat.random(n_trn, data_dim, density=0.8, format="csr", dtype=np.float32)
    X_tst = smat.random(n_tst, data_dim, density=0.8, format="csr", dtype=np.float32)
    Y_true = -X_tst.dot(X_trn.T).toarray()
    Y_true = np.argsort(Y_true)[:, :top_k]
    recall = run_one(X_trn, X_tst, Y_true)
    assert recall == approx(1.0, abs=1e-2), f"hnsw with data_type=csr failed, recall={recall}"


def test_hnsw_predict():
    import random
    import numpy as np
    import scipy.sparse as smat
    from pecos.ann import HNSW

    random.seed(1234)
    np.random.seed(1234)
    top_k = 10
    M, efC, efS = 32, 100, 50
    max_level, threads = 5, 16
    n_trn, n_tst = 90, 10
    metric_type = "l2"
    num_searcher_batch = 4
    num_searcher_online = 1

    def run_one(X_trn, X_tst):
        model = HNSW.train(X_trn, M, efC, max_level, metric_type, threads)
        Y_pred = model.predict(
            X_tst, efS, top_k, threads=num_searcher_batch, searchers=None, ret_csr=True
        )
        return model, Y_pred

    # test drm
    data_dim = 2
    X_trn = np.random.uniform(low=0.0, high=1.0, size=(n_trn, data_dim)).astype(np.float32)
    X_tst = np.random.uniform(low=0.0, high=1.0, size=(n_tst, data_dim)).astype(np.float32)
    model, Yp_batch = run_one(X_trn, X_tst)
    searchers = model.searchers_create(num_searcher_online)
    Yp_online = model.predict(
        X_tst, efS, top_k, threads=num_searcher_online, searchers=searchers, ret_csr=True
    )
    assert Yp_batch.toarray() == approx(
        Yp_online.toarray()
    ), f"for data_type=drm, Yp_batch(thread={num_searcher_batch}) NOT CONSISTENT with Yp_online(thread={num_searcher_online})"

    # test csr
    data_dim = 50
    X_trn = smat.random(n_trn, data_dim, density=0.8, format="csr", dtype=np.float32)
    X_tst = smat.random(n_tst, data_dim, density=0.8, format="csr", dtype=np.float32)
    model, Yp_batch = run_one(X_trn, X_tst)
    searchers = model.searchers_create(num_searcher_online)
    Yp_online = model.predict(
        X_tst, efS, top_k, threads=num_searcher_online, searchers=searchers, ret_csr=True
    )
    assert Yp_batch.toarray() == approx(
        Yp_online.toarray()
    ), f"for data_type=csr, Yp_batch(thread={num_searcher_batch}) NOT CONSISTENT with Yp_online(thread={num_searcher_online})"

    # delete searchers and indexer
    del searchers, model
