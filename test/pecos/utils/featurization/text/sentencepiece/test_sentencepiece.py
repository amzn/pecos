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
import shlex
import subprocess

from pecos.utils.featurization.text.sentencepiece import SentencePieceModel as SPM


def test_importable():
    import pecos.utils.featurization.text.sentencepiece  # noqa: F401
    from pecos.utils.featurization.text import sentencepiece  # noqa: F401
    from pecos.utils.featurization.text.sentencepiece import SentencePieceModel  # noqa: F401


train_files = [
    "test/tst-data/utils/data/sentencepiece_train.src",
    "test/tst-data/utils/data/sentencepiece_train.tgt",
]


def test_word_model(tmpdir):
    model = SPM.train(train_files, model_type="word", vocab_size="110")
    predicted_file = tmpdir.join("predicted_file")
    ref_file = "test/tst-data/utils/data/sentencepiece_word_tokenized.src"
    model.predict_file(train_files[0], str(predicted_file), to_ids=False)
    with open(ref_file, "r", encoding="utf-8") as fref:
        assert predicted_file.read_text(encoding="utf-8") == fref.read()


def test_unigram_model(tmpdir):
    model = SPM.train(
        train_files,
        vocab_size=60,
        model_type="unigram",
        max_sentence_length=200,
        shuffle_input_sentence=False,
    )
    model.save(str(tmpdir.join("unigram").mkdir()))

    test_input = ["hello world"]
    encoded_ids = model.predict(test_input, to_ids=True)
    decoded_output = model.decode(encoded_ids, from_ids=True)
    assert decoded_output[0] == test_input[0]


def test_cli_bpe_model(tmpdir, capsys):

    test_input_file = tmpdir.join("test_input")
    test_input_file.write_text("hello world", encoding="utf-8")
    test_encoded_file = tmpdir.join("test_encoded")
    test_decoded_file = tmpdir.join("test_decoded")

    cmd = []
    cmd += ["python3 -m pecos.utils.featurization.text.sentencepiece.train"]
    cmd += ["-m {}".format(str(tmpdir))]
    cmd += ["-i {}".format(",".join(train_files))]
    cmd += ["-t bpe"]
    cmd += ["-v 60"]
    cmd += ["--max-sentence-length 30"]
    cmd += ["--shuffle-input-sentence yes"]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0

    cmd = []
    cmd += ["python3 -m pecos.utils.featurization.text.sentencepiece.predict"]
    cmd += ["-m {}".format(str(tmpdir))]
    cmd += ["-i {}".format(test_input_file)]
    cmd += ["-o {}".format(str(test_encoded_file))]
    cmd += ["--output-type pieces"]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0

    model = SPM.load(str(tmpdir))
    model.decode_file(str(test_encoded_file), str(test_decoded_file), from_ids=False)
    assert test_decoded_file.read_text(encoding="utf-8") == "hello world\n"

    model.decode_file(str(test_encoded_file), "-", from_ids=False)
    out, err = capsys.readouterr()
    assert out == "hello world\n"
