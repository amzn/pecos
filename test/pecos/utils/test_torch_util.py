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


def test_setup_device():
    import torch
    from pecos.utils import torch_util

    if torch.cuda.is_available():  # GPU machine
        device, n_active_gpu = torch_util.setup_device(use_gpu_if_available=True)
        assert device == torch.device("cuda")
        assert n_active_gpu == torch.cuda.device_count()
        device, n_active_gpu = torch_util.setup_device(use_gpu_if_available=False)
        assert device == torch.device("cpu")
        assert n_active_gpu == 0
    else:
        device, n_active_gpu = torch_util.setup_device(use_gpu_if_available=True)
        assert device == torch.device("cpu")
        assert n_active_gpu == 0
        device, n_active_gpu = torch_util.setup_device(use_gpu_if_available=False)
        assert device == torch.device("cpu")
        assert n_active_gpu == 0
