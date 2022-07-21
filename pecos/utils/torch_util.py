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
import logging

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def setup_device(use_gpu_if_available=True, device_id=-1):
    """Setup device for pytorch.

    Args:
        use_gpu_if_available (bool, optional): whether to use GPU if available. Default True
        device_id (int, optional): GPU id to use. Default -1 to use all

    Returns:
        device (torch.device): torch device
        n_active_gpu (int): number of GPUs available for torch.cuda
    """
    if use_gpu_if_available and torch.cuda.is_available():
        if device_id >= 0:
            # use specified device
            device = torch.device("cuda", device_id)
            n_active_gpu = 1
        else:
            # regular dataparallel
            device = torch.device("cuda")
            n_active_gpu = torch.cuda.device_count()
    else:
        if use_gpu_if_available:
            LOGGER.warning("CUDA is not available, will fall back to CPU.")
        if torch.cuda.is_available():
            LOGGER.warning("CUDA is available but disabled, will only use CPU.")
        device = torch.device("cpu")
        n_active_gpu = 0
    LOGGER.info(f"Setting device to {device}, number of active GPUs: {n_active_gpu}")
    return device, n_active_gpu


def set_seed(seed=0):
    """Set the random seed for torch.

    Args:
        seet (int, optional): random seed. Default 0
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If CUDA is not available, this is silently ignored.
    torch.cuda.manual_seed_all(seed)


def apply_mask(hidden_states, masks):
    """Apply mask to masked tokens in a batch

    Args:
        hidden_states (torch.tensor): shape(batch_size, seq_len(PADDED), hidden_dim)
        masks (torch.tensor): shape(batch_size, seq_len(PADDED)) where masked tokens=0, unmasked tokens=1

    Returns:
        masked_hidden_states (torch.tensor): shape(batch_size, seq_len(PADDED), hidden_dim)
    """
    hidden_dim = hidden_states.shape[-1]
    hidden_states.view(-1, hidden_dim)[~masks.view(-1).type(torch.ByteTensor), :] = 0
    return hidden_states
