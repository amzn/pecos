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
    import pecos.xmr  # noqa: F401
    import pecos.xmr.reranker  # noqa: F401
    from pecos.xmr.reranker.model import TextNumrEncoder  # noqa: F401
    from pecos.xmr.reranker.model import RankingModel  # noqa: F401
    from pecos.xmr.reranker.trainer import RankingTrainer  # noqa: F401


def test_numr_encoder():
    import torch
    from pecos.xmr.reranker.model import NumrMLPEncoderConfig
    from pecos.xmr.reranker.model import NumrMLPEncoder

    numr_config = NumrMLPEncoderConfig(
        inp_feat_dim=2,
        inp_dropout_prob=0.0,
        hid_dropout_prob=0.0,
        hid_actv_type="identity",
        hid_size_list=[2],
    )
    assert numr_config.inp_feat_dim == 2
    assert numr_config.inp_dropout_prob == 0.0
    assert numr_config.hid_dropout_prob == 0.0
    assert numr_config.hid_actv_type == "identity"
    assert numr_config.hid_size_list == [2]

    numr_encoder = NumrMLPEncoder(numr_config)
    linear_layer = numr_encoder.mlp_block.mlp_layers[0]
    linear_layer.bias.data.fill_(0.0)
    linear_layer.weight.data.fill_(0.0)
    linear_layer.weight.data.fill_diagonal_(1.0)
    with torch.no_grad():
        inp_feat = torch.tensor([-1, 1]).float()
        out_feat = numr_encoder(inp_feat)
    assert out_feat.numpy() == approx(
        out_feat.numpy(),
        abs=0.0,
    ), f"Enc(inp_feat) != inp_feat, given Enc is identity"


def test_textnumr_encoder():
    import torch
    from transformers import set_seed
    from transformers import AutoConfig, AutoTokenizer
    from pecos.xmr.reranker.model import TextNumrEncoderConfig
    from pecos.xmr.reranker.model import TextNumrEncoder

    enc_list = [
        "prajjwal1/bert-tiny",
        "sentence-transformers/all-MiniLM-L6-v2",
        "intfloat/multilingual-e5-small",
    ]
    ans_list = [
        0.007879042997956276,
        0.0035168465692549944,
        -0.0047034271992743015,
    ]
    set_seed(1234)

    for idx, enc_name in enumerate(enc_list):
        text_config = AutoConfig.from_pretrained(
            enc_name,
            hidden_dropout_prob=0.0,
        )
        textnumr_config = TextNumrEncoderConfig(
            text_config=text_config,
            numr_config=None,
            text_pooling_type="cls",
            head_actv_type="identity",
            head_dropout_prob=0.0,
            head_size_list=[1],
        )
        textnumr_encoder = TextNumrEncoder(textnumr_config)
        linear_layer = textnumr_encoder.head_layers.mlp_layers[0]
        linear_layer.bias.data.fill_(0.0)
        linear_layer.weight.data.fill_(0.0)
        linear_layer.weight.data.fill_diagonal_(1.0)
        textnumr_encoder.scorer.bias.data.fill_(0.0)
        textnumr_encoder.scorer.weight.data.fill_(1.0)

        # obtained from bert-tiny tokenizer("I Like coffee")
        tokenizer = AutoTokenizer.from_pretrained(enc_name)
        input_dict = tokenizer("I Like coffee", return_tensors="pt")
        outputs = textnumr_encoder(**input_dict)
        assert outputs.text_emb is not None
        assert outputs.numr_emb is None

        text_emb = outputs.text_emb
        mu = torch.mean(text_emb).item()
        assert mu == approx(
            ans_list[idx],
            abs=1e-3,
        ), f"mu(text_emb)={mu} != {ans_list[idx]}"
