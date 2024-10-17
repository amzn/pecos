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
    import pecos.xmr  # noqa: F401
    import pecos.xmr.reranker  # noqa: F401
    from pecos.xmr.reranker.model import TextNumrEncoder  # noqa: F401
    from pecos.xmr.reranker.model import RankingModel  # noqa: F401
    from pecos.xmr.reranker.trainer import RankingTrainer  # noqa: F401


def test_model():
    from pecos.xmr.reranker.model import NumrMLPEncoderConfig

    mlp_config = NumrMLPEncoderConfig(
        inp_feat_dim=5,
        inp_dropout_prob=0.5,
        hid_actv_type="gelu",
        hid_size_list=[8, 16],
    )
    assert mlp_config.inp_feat_dim == 5
    assert mlp_config.inp_dropout_prob == 0.5
    assert mlp_config.hid_actv_type == "gelu"
    assert mlp_config.hid_size_list == [8, 16]
