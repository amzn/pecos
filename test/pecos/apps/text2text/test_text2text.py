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


def test_importable():
    import pecos.apps.text2text  # noqa: F401
    from pecos.apps import text2text  # noqa: F401
    from pecos.apps.text2text import Text2Text  # noqa: F401


def test_cli(tmpdir):
    import subprocess
    import shlex
    import json

    def assert_json_string(str_a, str_b):
        json_a = json.loads(str_a)
        json_b = json.loads(str_b)
        json_a["schema"] == json_b["schema"]
        sorted_output_a = sorted(json_a["data"], key=lambda x: (-x[1], x[0]))
        sorted_output_b = sorted(json_b["data"], key=lambda x: (-x[1], x[0]))
        for aa, bb in zip(sorted_output_a, sorted_output_b):
            assert aa[0] == bb[0]
            assert pytest.approx(aa[1], 1e-6) == bb[1]

    train_file = "test/tst-data/apps/text2text/train.txt"
    train_with_rel_file = "test/tst-data/apps/text2text/train_with_rel.txt"
    test_file = "test/tst-data/apps/text2text/test.txt"
    item_file = "test/tst-data/apps/text2text/query_text.txt"
    item_to_keep_file = "test/tst-data/apps/text2text/query_to_keep_text.txt"
    sklearn_tfidf_true_pred_file = "test/tst-data/apps/text2text/sklearn_tfidf_true_pred_items.txt"
    true_pred_file = "test/tst-data/apps/text2text/true_pred_items.txt"
    true_pred_keep_file = "test/tst-data/apps/text2text/sklearn_tfidf_true_pred_items_keep.txt"
    truth_format1_file = "test/tst-data/apps/text2text/truth_items.txt"
    truth_format2_file = "test/tst-data/apps/text2text/test.txt"

    test_pred_file = str(tmpdir.join("pred_items.txt"))
    test_pred_keep_file = str(tmpdir.join("pred_items_keep.txt"))
    model_folder = str(tmpdir.join("save_model"))

    # Training with SklearnTfidf
    cmd = []
    cmd += ["python3 -m pecos.apps.text2text.train"]
    cmd += ["-i {}".format(train_file)]
    cmd += ["-q {}".format(item_file)]
    cmd += ["-m {}".format(model_folder)]
    cmd += ["--max-leaf-size {}".format(10)]
    cmd += ["""--vectorizer-config-json '{"type":"sklearntfidf","kwargs":{}}'"""]
    print(" ".join(cmd))
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0

    # Batch Inference
    cmd = []
    cmd += ["python3 -m pecos.apps.text2text.predict"]
    cmd += ["-i {}".format(test_file)]
    cmd += ["-m {}".format(model_folder)]
    cmd += ["-o {}".format(test_pred_file)]
    cmd += ["-T 0.001"]
    cmd += ["--meta-info-path {}".format(test_file)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(" ".join(cmd))
    assert process.returncode == 0
    for line_test, line_true in zip(
        open(test_pred_file, "r", encoding="utf-8"),
        open(sklearn_tfidf_true_pred_file, "r", encoding="utf-8"),
    ):
        assert_json_string(line_test, line_true)

    # Batch Inference with subset of keep-outputs
    cmd = []
    cmd += ["python3 -m pecos.apps.text2text.predict"]
    cmd += ["-i {}".format(test_file)]
    cmd += ["-m {}".format(model_folder)]
    cmd += ["-o {}".format(test_pred_keep_file)]
    cmd += ["--output-to-keep-path {}".format(item_to_keep_file)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(" ".join(cmd))
    assert process.returncode == 0
    for line_test, line_true in zip(
        open(test_pred_keep_file, "r", encoding="utf-8"),
        open(true_pred_keep_file, "r", encoding="utf-8"),
    ):
        assert_json_string(line_test, line_true)

    # Training with Tfidf
    cmd = []
    cmd += ["python3 -m pecos.apps.text2text.train"]
    cmd += ["-i {}".format(train_file)]
    cmd += ["-q {}".format(item_file)]
    cmd += ["-m {}".format(model_folder)]
    cmd += ["--max-leaf-size {}".format(10)]
    print(" ".join(cmd))
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0

    # Batch Inference
    cmd = []
    cmd += ["python3 -m pecos.apps.text2text.predict"]
    cmd += ["-i {}".format(test_file)]
    cmd += ["-m {}".format(model_folder)]
    cmd += ["-o {}".format(test_pred_file)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(" ".join(cmd))
    assert process.returncode == 0
    for line_test, line_true in zip(
        open(test_pred_file, "r", encoding="utf-8"), open(true_pred_file, "r", encoding="utf-8")
    ):
        assert_json_string(line_test, line_true)

    # Realtime Inference
    cmd = []
    cmd += ["python3", "-m", "pecos.apps.text2text.predict"]
    cmd += ["-m", model_folder]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    with open(test_file, "r", encoding="utf-8") as fin, open(
        true_pred_file, "r", encoding="utf-8"
    ) as f_expect:
        input = "\n".join([tt.strip().split("\t")[1] for tt in fin.readlines()])
        test_output = (
            process.communicate(input=input.encode("utf-8"))[0].decode("utf-8").strip("\n")
        )
        print(input)
        print(test_output)
        expected_output = f_expect.read().strip("\n")
        for test_string, expected_string in zip(
            test_output.split("\n"), expected_output.split("\n")
        ):
            assert_json_string(test_string, expected_string)
    assert process.returncode == 0

    std_output = b"==== evaluation results ====\nprec   = 0.00 0.00 33.33 25.00 20.00 16.67 14.29 12.50 11.11 10.00\nrecall = 0.00 0.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00\n"

    # For ground truth file with Format 1
    cmd = []
    cmd += ["python3 -m pecos.apps.text2text.evaluate"]
    cmd += ["-p {}".format(sklearn_tfidf_true_pred_file)]
    cmd += ["-y {}".format(truth_format1_file)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(" ".join(cmd))
    assert process.returncode == 0
    assert process.stdout == std_output

    # For ground truth file with Format 2
    cmd = []
    cmd += ["python3 -m pecos.apps.text2text.evaluate"]
    cmd += ["-p {}".format(sklearn_tfidf_true_pred_file)]
    cmd += ["-y {}".format(truth_format2_file)]
    cmd += ["-q {}".format(item_file)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(" ".join(cmd))
    assert process.returncode == 0
    assert process.stdout == std_output

    # Training for testing the workspace folder that stores intermediate variables
    # First-time train and store intermediate variables in workspace_folder
    workspace_folder = str(tmpdir.join("tmp_ws"))
    cmd = []
    cmd += ["python3 -m pecos.apps.text2text.train"]
    cmd += ["-i {}".format(train_file)]
    cmd += ["-q {}".format(item_file)]
    cmd += ["-m {}".format(model_folder)]
    cmd += ["--max-leaf-size {}".format(10)]
    cmd += ["--workspace-folder {}".format(workspace_folder)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(" ".join(cmd))
    assert process.returncode == 0

    # Second-time train and load intermediate variables from workspace_folder
    cmd = []
    cmd += ["python3 -m pecos.apps.text2text.train"]
    cmd += ["-i {}".format(train_file)]
    cmd += ["-q {}".format(item_file)]
    cmd += ["-m {}".format(model_folder)]
    cmd += ["--workspace-folder {}".format(workspace_folder)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(" ".join(cmd))
    assert process.returncode == 0

    # Training with pifa_lf_concat
    Z_pifa_file = "test/tst-data/apps/text2text/Z.pifa.npz"
    cmd = []
    cmd += ["python3 -m pecos.apps.text2text.train"]
    cmd += ["-i {}".format(train_file)]
    cmd += ["-q {}".format(item_file)]
    cmd += ["-m {}".format(model_folder)]
    cmd += ["--max-leaf-size {}".format(10)]
    cmd += ["--label-embed-type pifa_lf_concat::Z={}".format(Z_pifa_file)]
    print(" ".join(cmd))
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0

    cmd = []
    cmd += ["python3 -m pecos.apps.text2text.predict"]
    cmd += ["-i {}".format(test_file)]
    cmd += ["-m {}".format(model_folder)]
    cmd += ["-o {}".format(test_pred_file)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(" ".join(cmd))
    assert process.returncode == 0
    for line_test, line_true in zip(
        open(test_pred_file, "r", encoding="utf-8"), open(true_pred_file, "r", encoding="utf-8")
    ):
        assert_json_string(line_test, line_true)

    # train with cost sensitive in OVA setting
    rel_model_folder = str(tmpdir.join("save_model_rel"))
    rel_pred_file = str(tmpdir.join("pred_items_rel.txt"))

    cmd = []
    cmd += ["python3 -m pecos.apps.text2text.train"]
    cmd += ["-i {}".format(train_with_rel_file)]
    cmd += ["-q {}".format(item_file)]
    cmd += ["-m {}".format(rel_model_folder)]
    cmd += ["--rel-norm no-norm"]
    cmd += ["--rel-mode induce"]
    print(" ".join(cmd))
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0

    cmd = []
    cmd += ["python3 -m pecos.apps.text2text.predict"]
    cmd += ["-i {}".format(test_file)]
    cmd += ["-m {}".format(rel_model_folder)]
    cmd += ["-o {}".format(rel_pred_file)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(" ".join(cmd))
    assert process.returncode == 0

    cp2_model_folder = str(tmpdir.join("save_model_cp2"))
    cp2_pred_file = str(tmpdir.join("pred_items_cp2.txt"))

    cmd = []
    cmd += ["python3 -m pecos.apps.text2text.train"]
    cmd += ["-i {}".format(train_file)]
    cmd += ["-q {}".format(item_file)]
    cmd += ["-m {}".format(cp2_model_folder)]
    cmd += ["--Cp {}".format(2.0)]
    print(" ".join(cmd))
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0

    # Batch Inference
    cmd = []
    cmd += ["python3 -m pecos.apps.text2text.predict"]
    cmd += ["-i {}".format(test_file)]
    cmd += ["-m {}".format(cp2_model_folder)]
    cmd += ["-o {}".format(cp2_pred_file)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(" ".join(cmd))
    assert process.returncode == 0
    for line_test, line_true in zip(
        open(rel_pred_file, "r", encoding="utf-8"), open(cp2_pred_file, "r", encoding="utf-8")
    ):
        assert_json_string(line_test, line_true)
