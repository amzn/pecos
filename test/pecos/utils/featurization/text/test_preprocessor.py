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

import numpy as np
import scipy.sparse as smat
from pecos.utils import smat_util
from pytest import approx

src_input_file = "test/tst-data/utils/data/train.src"
src_text_input_file = "test/tst-data/utils/data/train_text.src"
label_file = "test/tst-data/utils/data/label_vocab.txt"
tfidf_vectorizer_config_path = "test/tst-data/utils/data/tfidf_vectorizer_config.json"
tfidf_tgt_input_file = "test/tst-data/utils/data/train.tfidf.tgt.npz"
sklearn_tfidf_vectorizer_config_path = (
    "test/tst-data/utils/data/sklearn_tfidf_vectorizer_config.json"
)
sklearn_tfidf_tgt_input_file = "test/tst-data/utils/data/train.sklearn_tfidf.tgt.npz"
hashing_vectorizer_config_path = "test/tst-data/utils/data/hashing_vectorizer_config.json"
hashing_tgt_input_file = "test/tst-data/utils/data/train.hashing.tgt.npz"


def assert_matrix_equal(Xtgt, X, tolerance=1e-6):
    assert type(X) == type(Xtgt)
    if isinstance(Xtgt, np.ndarray):
        assert Xtgt == approx(X, abs=tolerance)
    elif isinstance(Xtgt, smat.spmatrix):
        assert Xtgt.todense() == approx(X.todense(), abs=tolerance)
    else:
        raise ValueError("Un recognized matrix format {}".format(type(X)))


def preprocessor_cli(tmpdir, config_path, tgt_input_file):
    import subprocess
    import shlex

    model_folder = str(tmpdir.join("vectorizer"))
    x_file = str(tmpdir.join("x"))
    y_file = str(tmpdir.join("y.npz"))

    # Build
    cmd = []
    cmd += ["python3 -m pecos.utils.featurization.text.preprocess"]
    cmd += ["build"]
    cmd += ["-i {}".format(src_input_file)]
    cmd += ["--text-pos 1"]
    cmd += ["--vectorizer-config-path {}".format(config_path)]
    cmd += ["-m {}".format(model_folder)]
    print(" ".join(cmd))
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0

    # Run
    cmd = []
    cmd += ["python3 -m pecos.utils.featurization.text.preprocess"]
    cmd += ["run"]
    cmd += ["-i {}".format(src_input_file)]
    cmd += ["-l {}".format(label_file)]
    cmd += ["-p {}".format(model_folder)]
    cmd += ["-x {}".format(x_file)]
    cmd += ["-y {}".format(y_file)]
    cmd += ["--text-pos 1"]
    cmd += ["--label-pos 0"]
    cmd += ["--threads 1"]
    print(" ".join(cmd))
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0
    X = smat_util.load_matrix(x_file)
    Xtgt = smat_util.load_matrix(tgt_input_file)
    assert_matrix_equal(Xtgt, X)

    # Run without labels
    cmd = []
    cmd += ["python3 -m pecos.utils.featurization.text.preprocess"]
    cmd += ["run"]
    cmd += ["-i {}".format(src_input_file)]
    cmd += ["-p {}".format(model_folder)]
    cmd += ["-x {}".format(x_file)]
    cmd += ["--text-pos 1"]
    print(" ".join(cmd))
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0
    X = smat_util.load_matrix(x_file)
    Xtgt = smat_util.load_matrix(tgt_input_file)
    assert_matrix_equal(Xtgt, X)


def preprocessor_from_file_cli(tmpdir, config_path, tgt_input_file):
    import subprocess
    import shlex

    model_folder = str(tmpdir.join("vectorizer"))
    x_file = str(tmpdir.join("x"))

    # Build
    cmd = []
    cmd += ["python3 -m pecos.utils.featurization.text.preprocess"]
    cmd += ["build"]
    cmd += ["-i {}".format(src_text_input_file)]
    cmd += ["--text-pos 0"]
    cmd += ["--from-file true"]
    cmd += ["--vectorizer-config-path {}".format(config_path)]
    cmd += ["-m {}".format(model_folder)]
    print(" ".join(cmd))
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0

    # Run without labels
    cmd = []
    cmd += ["python3 -m pecos.utils.featurization.text.preprocess"]
    cmd += ["run"]
    cmd += ["-i {}".format(src_text_input_file)]
    cmd += ["-p {}".format(model_folder)]
    cmd += ["-x {}".format(x_file)]
    cmd += ["--text-pos 0"]
    cmd += ["--from-file true"]
    print(" ".join(cmd))
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0
    X = smat_util.load_matrix(x_file)
    Xtgt = smat_util.load_matrix(tgt_input_file)
    assert_matrix_equal(Xtgt, X)


def test_tfidf_vectorizer(tmpdir):
    preprocessor_cli(tmpdir, tfidf_vectorizer_config_path, tfidf_tgt_input_file)
    preprocessor_from_file_cli(tmpdir, tfidf_vectorizer_config_path, tfidf_tgt_input_file)


def test_sklearn_tfidf_vectorizer(tmpdir):
    preprocessor_cli(tmpdir, sklearn_tfidf_vectorizer_config_path, sklearn_tfidf_tgt_input_file)


def test_hashing_vectorizer(tmpdir):
    preprocessor_cli(tmpdir, hashing_vectorizer_config_path, hashing_tgt_input_file)
