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


def test_save_and_load(tmpdir):
    import random
    import numpy as np
    from pecos.ann.hnsw import HNSW
    from pecos.utils import smat_util

    random.seed(1234)
    np.random.seed(1234)
    X_trn = smat_util.load_matrix("test/tst-data/ann/X.trn.l2-normalized.npy").astype(np.float32)
    X_tst = smat_util.load_matrix("test/tst-data/ann/X.tst.l2-normalized.npy").astype(np.float32)
    model_folder = tmpdir.join("hnsw_model_dir")

    train_params = HNSW.TrainParams(M=36, efC=90, metric_type="ip", threads=1)
    pred_params = HNSW.PredParams(efS=80, topk=10, threads=1)
    model = HNSW.train(
        X_trn,
        train_params=train_params,
        pred_params=pred_params,
    )
    Yp_from_mem, _ = model.predict(X_tst, ret_csr=False)
    model.save(model_folder)
    del model

    model = HNSW.load(model_folder)
    Yp_from_file, _ = model.predict(X_tst, pred_params=pred_params, ret_csr=False)
    assert Yp_from_mem == approx(
        Yp_from_file, abs=0.0
    ), f"save and load failed: Yp_from_mem != Yp_from_file"


def test_predict_and_recall():
    import random
    import numpy as np
    import scipy.sparse as smat
    from pecos.utils import smat_util
    from pecos.ann.hnsw import HNSW

    random.seed(1234)
    np.random.seed(1234)
    top_k = 10
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
    dense_model_folder = "test/tst-data/ann/hnsw-model-dense"
    sparse_model_folder = "test/tst-data/ann/hnsw-model-sparse"

    # compute exact NN ground truth
    # for both ip and cosine similarity, since data is l2-normalized
    Y_true = 1.0 - X_tst.dot(X_trn.T)
    Y_true = np.argsort(Y_true)[:, :top_k]

    # test dense features
    model = HNSW.load(dense_model_folder)
    searchers = model.searchers_create(num_searcher_online)
    pred_params = model.get_pred_params()
    for efS in efS_list:
        pred_params.efS = efS
        Y_pred, _ = model.predict(
            X_tst, pred_params=pred_params, searchers=searchers, ret_csr=False
        )
        recall = calc_recall(Y_true, Y_pred)
        assert recall == approx(
            1.0, abs=1e-2
        ), f"hnsw inference failed: data_type=drm, efS={efS}, recall={recall}"
    del searchers, model

    # test csr features, we just reuse the Y_true since data are the same
    X_trn = smat.csr_matrix(X_trn).astype(np.float32)
    X_tst = smat.csr_matrix(X_tst).astype(np.float32)
    model = HNSW.load(sparse_model_folder)
    searchers = model.searchers_create(num_searcher_online)
    pred_params = model.get_pred_params()
    for efS in efS_list:
        pred_params.efS = efS
        Y_pred, _ = model.predict(
            X_tst, pred_params=pred_params, searchers=searchers, ret_csr=False
        )
        recall = calc_recall(Y_true, Y_pred)
        assert recall == approx(
            1.0, abs=1e-2
        ), f"hnsw inference failed: data_type=csr, efS={efS}, recall={recall}"
    del searchers, model


def test_cli(tmpdir):
    import subprocess
    import shlex

    x_trn_path = "test/tst-data/ann/X.trn.l2-normalized.npy"
    x_tst_path = "test/tst-data/ann/X.tst.l2-normalized.npy"
    model_folder = str(tmpdir.join("hnsw_save_model"))
    y_pred_path = str(tmpdir.join("Yt_pred.npz"))

    # train
    cmd = []
    cmd += ["python3 -m pecos.ann.hnsw.train"]
    cmd += ["-x {}".format(x_trn_path)]
    cmd += ["-m {}".format(model_folder)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)

    # predict
    cmd = []
    cmd += ["python3 -m pecos.ann.hnsw.predict"]
    cmd += ["-x {}".format(x_tst_path)]
    cmd += ["-m {}".format(model_folder)]
    cmd += ["-o {}".format(y_pred_path)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)
