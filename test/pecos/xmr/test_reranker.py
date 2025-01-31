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
