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
import numpy as np

import pytest  # noqa: F401; pylint: disable=unused-variable

bert_model_path = "test/tst-data/xmc/xtransformer/saved_model/"
train_feat_file = "test/tst-data/xmc/xtransformer/train_feat.npz"
train_dense_feat_file = "test/tst-data/xmc/xtransformer/dense_train_feat.npy"


def test_bert(tmpdir):
    from pecos.utils import torch_util

    _, n_gpu = torch_util.setup_device()
    # test on CPU
    xtransformer_cli(tmpdir.join("sparse_cpu"), bert_model_path, train_feat_file, 0)
    xtransformer_cli(tmpdir.join("dense_cpu"), bert_model_path, train_dense_feat_file, 0)

    if n_gpu > 0:
        # test on all GPUs
        xtransformer_cli(tmpdir.join("sparse_gpu"), bert_model_path, train_feat_file, n_gpu)
        xtransformer_cli(tmpdir.join("dense_gpu"), bert_model_path, train_dense_feat_file, n_gpu)

    if n_gpu > 1:
        # test on single GPU when multi-GPU available
        xtransformer_cli(tmpdir.join("sparse_single_gpu"), bert_model_path, train_feat_file, 1)
        xtransformer_cli(tmpdir.join("dense_single_gpu"), bert_model_path, train_dense_feat_file, 1)


def xtransformer_cli(tmpdir, load_model_path, X_feat_file, nr_gpus):
    import subprocess
    import shlex
    import os

    X_trn_file = "test/tst-data/xmc/xtransformer/train.txt"
    code_file = "test/tst-data/xmc/xtransformer/clusters.npz"
    Y_trn_file = "test/tst-data/xmc/xtransformer/train_label.npz"
    save_P_file = str(tmpdir.join("P.npz"))

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in range(nr_gpus)])

    # Training matcher
    cmd = []
    cmd += ["python3 -m pecos.xmc.xtransformer.train"]
    cmd += ["--trn-feat-path {}".format(X_feat_file)]
    cmd += ["--trn-label-path {}".format(Y_trn_file)]
    cmd += ["--model-dir {}".format(str(tmpdir))]
    cmd += ["--code-path {}".format(code_file)]
    cmd += ["--trn-text-path {}".format(X_trn_file)]
    cmd += ["--init-model-dir {}".format(load_model_path)]
    cmd += ["--batch-size {}".format(1)]
    cmd += ["--num-train-epochs {}".format(1)]
    cmd += ["--save-steps {}".format(2)]
    cmd += ["--only-topk {}".format(2)]
    cmd += ["--batch-gen-workers {}".format(2)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)

    # Training matcher with indexing
    cmd = []
    cmd += ["python3 -m pecos.xmc.xtransformer.train"]
    cmd += ["--trn-feat-path {}".format(X_feat_file)]
    cmd += ["--trn-label-path {}".format(Y_trn_file)]
    cmd += ["--model-dir {}".format(str(tmpdir))]
    cmd += ["--trn-text-path {}".format(X_trn_file)]
    cmd += ["--init-model-dir {}".format(load_model_path)]
    cmd += ["--batch-size {}".format(1)]
    cmd += ["--num-train-epochs {}".format(1)]
    cmd += ["--save-steps {}".format(2)]
    cmd += ["--only-topk {}".format(2)]
    cmd += ["--batch-gen-workers {}".format(2)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)

    # Predict
    cmd = []
    cmd += ["python3 -m pecos.xmc.xtransformer.predict"]
    cmd += ["--feat-path {}".format(X_feat_file)]
    cmd += ["--text-path {}".format(X_trn_file)]
    cmd += ["--model-folder {}".format(str(tmpdir))]
    cmd += ["--output-dir {}".format(str(tmpdir))]
    cmd += ["--batch-size {}".format(1)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)

    # predict with no batching while encoding
    second_predict_tmpdir = tmpdir.mkdir("second_predict")
    save_P_file2 = str(second_predict_tmpdir.join("P.npz"))
    # Predict
    cmd = []
    cmd += ["python3 -m pecos.xmc.xtransformer.predict"]
    cmd += ["--feat-path {}".format(X_feat_file)]
    cmd += ["--text-path {}".format(X_trn_file)]
    cmd += ["--model-folder {}".format(str(tmpdir))]
    cmd += ["--output-dir {}".format(str(second_predict_tmpdir))]
    cmd += ["--batch-gen-workers {0}".format(1)]
    cmd += ["--batch-size {}".format(1)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)

    first_predict = np.load(save_P_file)
    second_predict = np.load(save_P_file2)

    for key in first_predict.keys():
        if key != "format":
            assert np.allclose(
                first_predict[key], second_predict[key]
            ), "batched encoding and single encoding gives different results"

    # Evaluate ranker prediction
    cmd = []
    cmd += ["python3 -m pecos.xmc.xlinear.evaluate"]
    cmd += ["-y {}".format(Y_trn_file)]
    cmd += ["-p {}".format(save_P_file)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(process.stdout)
    assert process.returncode == 0, " ".join(cmd)
    std_output = b"==== evaluation results ====\nprec   = 100.00 100.00 66.67 50.00 40.00 33.33 28.57 25.00 22.22 20.00\nrecall = 41.67 83.33 83.33 83.33 83.33 83.33 83.33 83.33 83.33 83.33\n"
    assert process.stdout == std_output

    del os.environ["CUDA_VISIBLE_DEVICES"]
