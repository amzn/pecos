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
    from pecos.ann.hnsw import HNSW  # noqa: F401


def test_predict_and_recall():
    import random
    import numpy as np
    import scipy.sparse as smat
    from pecos.utils import smat_util
    from pecos.ann.hnsw import HNSW

    random.seed(1234)
    np.random.seed(1234)
    M, efC, top_k = 32, 100, 10
    max_level_upper_bound, threads = 5, 8
    efS_list = [50, 75, 100]
    num_searcher_online = 2

    def calc_recall(Y_true, Y_pred):
        n_data, top_k = Y_true.shape
        recall = 0.0
        for qid in range(n_data):
            yt = set(Y_true[qid, :].flatten().data)
            yp = set(Y_pred[qid, :].flatten().data)
            recall += len(yt.intersection(yp)) / top_k
        recall = recall / n_data
        return recall

    # load data matrices
    X_trn = smat_util.load_matrix("test/tst-data/ann/X.trn.l2-normalized.npy").astype(np.float32)
    X_tst = smat_util.load_matrix("test/tst-data/ann/X.tst.l2-normalized.npy").astype(np.float32)

    # compute exact NN ground truth
    # for both ip and cosine similarity, since data is l2-normalized
    metric_type = "ip"
    Y_true = 1.0 - X_tst.dot(X_trn.T)
    Y_true = np.argsort(Y_true)[:, :top_k]

    # test dense features
    model = HNSW.train(
        X_trn,
        M=M,
        efC=efC,
        max_level_upper_bound=max_level_upper_bound,
        metric_type=metric_type,
        threads=threads,
    )
    searchers = model.searchers_create(num_searcher_online)
    for efS in efS_list:
        Y_pred, _ = model.predict(X_tst, efS, top_k, searchers=searchers, ret_csr=False)
        recall = calc_recall(Y_true, Y_pred)
        assert recall == approx(
            1.0, abs=1e-2
        ), f"hnsw with data_type=drm failed: efS={efS}, recall={recall}"
    del searchers, model

    # test csr features, we just reuse the Y_true since data are the same
    X_trn = smat.csr_matrix(X_trn).astype(np.float32)
    X_tst = smat.csr_matrix(X_tst).astype(np.float32)

    model = HNSW.train(
        X_trn,
        M=M,
        efC=efC,
        max_level_upper_bound=max_level_upper_bound,
        metric_type=metric_type,
        threads=threads,
    )
    searchers = model.searchers_create(num_searcher_online)
    for efS in efS_list:
        Y_pred, _ = model.predict(X_tst, efS, top_k, searchers=searchers, ret_csr=False)
        recall = calc_recall(Y_true, Y_pred)
        assert recall == approx(
            1.0, abs=1e-2
        ), f"hnsw with data_type=csr failed: efS={efS}, recall={recall}"
    del searchers, model
