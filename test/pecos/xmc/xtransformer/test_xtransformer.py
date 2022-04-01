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
from pytest import approx

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
    save_P_file2 = str(tmpdir.join("P2.npz"))

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
    cmd += ["--save-pred-path {}".format(str(save_P_file))]
    cmd += ["--batch-size {}".format(1)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)

    # predict with max_pred_chunk=1
    cmd = []
    cmd += ["python3 -m pecos.xmc.xtransformer.predict"]
    cmd += ["--feat-path {}".format(X_feat_file)]
    cmd += ["--text-path {}".format(X_trn_file)]
    cmd += ["--model-folder {}".format(str(tmpdir))]
    cmd += ["--save-pred-path {}".format(str(save_P_file2))]
    cmd += ["--max-pred-chunk {}".format(1)]
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


def test_encode(tmpdir):
    import subprocess
    import shlex

    X_trn_file = "test/tst-data/xmc/xtransformer/train.txt"
    Y_trn_file = "test/tst-data/xmc/xtransformer/train_label.npz"

    model_folder = tmpdir.join("only_encoder")
    emb_path = model_folder.join("embeddings.npy")
    emb_path_B1 = model_folder.join("embeddings_B1.npy")

    # Training matcher
    cmd = []
    cmd += ["python3 -m pecos.xmc.xtransformer.train"]
    cmd += ["--trn-feat-path {}".format(train_feat_file)]
    cmd += ["--trn-label-path {}".format(Y_trn_file)]
    cmd += ["--model-dir {}".format(str(model_folder))]
    cmd += ["--trn-text-path {}".format(X_trn_file)]
    cmd += ["--init-model-dir {}".format(bert_model_path)]
    cmd += ["--batch-size {}".format(1)]
    cmd += ["--num-train-epochs {}".format(1)]
    cmd += ["--save-steps {}".format(2)]
    cmd += ["--only-topk {}".format(2)]
    cmd += ["--batch-gen-workers {}".format(2)]
    cmd += ["--only-encoder true"]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)

    # encode
    cmd = []
    cmd += ["python3 -m pecos.xmc.xtransformer.encode"]
    cmd += ["--text-path {}".format(X_trn_file)]
    cmd += ["--model-folder {}".format(str(model_folder))]
    cmd += ["--save-emb-path {}".format(str(emb_path))]
    cmd += ["--batch-gen-workers {}".format(1)]
    cmd += ["--batch-size {}".format(1)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)

    X_emb_pred = np.load(str(emb_path))

    # encode with max_pred_chunk=1
    cmd = []
    cmd += ["python3 -m pecos.xmc.xtransformer.encode"]
    cmd += ["--text-path {}".format(X_trn_file)]
    cmd += ["--model-folder {}".format(str(model_folder))]
    cmd += ["--save-emb-path {}".format(str(emb_path_B1))]
    cmd += ["--batch-gen-workers {}".format(1)]
    cmd += ["--batch-size {}".format(1)]
    cmd += ["--max-pred-chunk {}".format(1)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)

    X_emb_pred_B1 = np.load(str(emb_path_B1))
    assert X_emb_pred_B1 == approx(X_emb_pred, abs=1e-6)


def test_xtransformer_python_api():
    import numpy as np
    from pecos.utils import smat_util
    from pecos.utils.featurization.text.preprocess import Preprocessor

    from pecos.xmc.xtransformer.model import XTransformer
    from pecos.xmc.xtransformer.module import MLProblemWithText

    X_trn_file = "test/tst-data/xmc/xtransformer/train.txt"
    Y_trn_file = "test/tst-data/xmc/xtransformer/train_label.npz"

    trn_corpus = Preprocessor.load_data_from_file(
        X_trn_file,
        label_text_path=None,
        text_pos=0,
    )["corpus"]
    X_trn = smat_util.load_matrix(train_feat_file, dtype=np.float32)
    Y_trn = smat_util.load_matrix(Y_trn_file, dtype=np.float32)
    trn_prob = MLProblemWithText(trn_corpus, Y_trn, X_feat=X_trn)
    train_params = XTransformer.TrainParams.from_dict({}, recursive=True)

    train_params.matcher_params_chain.init_model_dir = bert_model_path
    train_params.matcher_params_chain.batch_size = 1
    train_params.matcher_params_chain.num_train_epochs = 1
    train_params.matcher_params_chain.save_steps = 2
    train_params.matcher_params_chain.batch_gen_workers = 2

    pred_params = XTransformer.PredParams.from_dict({}, recursive=True)
    pred_params.matcher_params_chain.only_topk = 2
    pred_params.ranker_params.hlm_args.model_chain.only_topk = 2

    print(train_params.to_dict())
    print(pred_params.to_dict())

    xtf = XTransformer.train(
        trn_prob,
        train_params=train_params,
        pred_params=pred_params,
    )
    P = xtf.predict(trn_corpus, X_trn)
    metric = smat_util.Metrics.generate(Y_trn, P, topk=10)
    std_output = "prec   = 100.00 100.00 66.67 50.00 40.00 33.33 28.57 25.00 22.22 20.00\nrecall = 41.67 83.33 83.33 83.33 83.33 83.33 83.33 83.33 83.33 83.33"
    assert str(metric) == std_output, f"{str(metric)} != {std_output}"


def test_xtransformer_param_skeleton(tmpdir):
    import subprocess
    import shlex
    import json
    from pecos.xmc.xtransformer.model import XTransformer

    X_trn_file = "test/tst-data/xmc/xtransformer/train.txt"
    Y_trn_file = "test/tst-data/xmc/xtransformer/train_label.npz"

    model_folder = tmpdir.join("skeleton")
    params_file = tmpdir.join("params.json")
    save_P_file = str(tmpdir.join("P.npz"))
    # gen train and pred skeleton to file
    cmd = []
    cmd += ["python3 -m pecos.xmc.xtransformer.train"]
    cmd += ["--generate-params-skeleton"]
    print(" ".join(cmd))
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)
    params = json.loads(process.stdout)
    train_params = XTransformer.TrainParams.from_dict(params["train_params"])
    pred_params = XTransformer.PredParams.from_dict(params["pred_params"])

    train_params.matcher_params_chain.init_model_dir = bert_model_path
    train_params.matcher_params_chain.batch_size = 1
    train_params.matcher_params_chain.num_train_epochs = 1
    train_params.matcher_params_chain.save_steps = 2
    train_params.matcher_params_chain.batch_gen_workers = 2

    pred_params.matcher_params_chain.only_topk = 2
    pred_params.ranker_params.hlm_args.model_chain.only_topk = 2

    params["train_params"] = train_params.to_dict()
    params["pred_params"] = pred_params.to_dict()
    print(json.dumps(params, indent=True))
    with open(str(params_file), "w") as fout:
        fout.write(json.dumps(params, indent=True))

    # Training matcher with indexing
    cmd = []
    cmd += ["python3 -m pecos.xmc.xtransformer.train"]
    cmd += ["--trn-feat-path {}".format(train_feat_file)]
    cmd += ["--trn-label-path {}".format(Y_trn_file)]
    cmd += ["--model-dir {}".format(str(model_folder))]
    cmd += ["--trn-text-path {}".format(X_trn_file)]
    cmd += ["--params-path {}".format(str(params_file))]
    cmd += ["--verbose-level {}".format(3)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(" ".join(cmd))
    assert process.returncode == 0, " ".join(cmd)

    # Predict
    cmd = []
    cmd += ["python3 -m pecos.xmc.xtransformer.predict"]
    cmd += ["--feat-path {}".format(train_feat_file)]
    cmd += ["--text-path {}".format(X_trn_file)]
    cmd += ["--model-folder {}".format(str(model_folder))]
    cmd += ["--save-pred-path {}".format(str(save_P_file))]
    cmd += ["--batch-size {}".format(1)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(" ".join(cmd))
    assert process.returncode == 0, " ".join(cmd)

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


def test_no_feat_mode(tmpdir):
    import subprocess
    import shlex

    X_trn_file = "test/tst-data/xmc/xtransformer/train.txt"
    Y_trn_file = "test/tst-data/xmc/xtransformer/train_label.npz"
    code_file = "test/tst-data/xmc/xtransformer/clusters.npz"

    model_folder = tmpdir.join("no_feat")
    save_P_file = model_folder.join("P_nofeat.npz")

    # Training matcher
    cmd = []
    cmd += ["python3 -m pecos.xmc.xtransformer.train"]
    cmd += ["--model-dir {}".format(str(model_folder))]
    cmd += ["--trn-label-path {}".format(Y_trn_file)]
    cmd += ["--trn-text-path {}".format(X_trn_file)]
    cmd += ["--init-model-dir {}".format(bert_model_path)]
    cmd += ["--code-path {}".format(code_file)]
    cmd += ["--batch-size {}".format(1)]
    cmd += ["--num-train-epochs {}".format(1)]
    cmd += ["--save-steps {}".format(2)]
    cmd += ["--only-topk {}".format(2)]
    cmd += ["--batch-gen-workers {}".format(2)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)

    # encode
    cmd = []
    cmd += ["python3 -m pecos.xmc.xtransformer.predict"]
    cmd += ["--text-path {}".format(X_trn_file)]
    cmd += ["--model-folder {}".format(str(model_folder))]
    cmd += ["--batch-gen-workers {}".format(1)]
    cmd += ["--batch-size {}".format(1)]
    cmd += ["--save-pred-path {}".format(str(save_P_file))]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)

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
    std_output = b"==== evaluation results ====\nprec   = 66.67 66.67 44.44 33.33 26.67 22.22 19.05 16.67 14.81 13.33\nrecall = 25.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00\n"
    assert process.stdout == std_output
