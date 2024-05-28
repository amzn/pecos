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
from typing import Union, Iterable

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


def clip_grad_norm_(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    r"""
    Implementation of torch.nn.utils.clip_grad_norm_ in torch==1.13
    This is to support sparse gradient with gradient clipping.
    REF: https://pytorch.org/docs/1.13/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    device = grads[0].device
    if norm_type == "inf":
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type
        )
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for g in grads:
        g.detach().mul_(clip_coef_clamped.to(g.device))
    return total_norm
