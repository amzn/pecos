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


def torch_sorted_csr_from_coo(shape, row_idx, col_idx, val, only_topk=None, device=None):
    """GPU-accelerated version of sorted_csr_from_coo using PyTorch.

    Returns a row-sorted CSR matrix from COO format. Each row's nonzero elements
    are sorted in descending order by value. Optimized for large datasets (10M+ rows).

    Args:
        shape (tuple): Shape of the output matrix (num_rows, num_cols)
        row_idx (ndarray or tensor): Row indices of COO matrix
        col_idx (ndarray or tensor): Column indices of COO matrix
        val (ndarray or tensor): Values of COO matrix
        only_topk (int, optional): Keep only top-k elements per row. Default None
        device (torch.device, optional): Device for computation. Default None (auto-detect)

    Returns:
        dict: Dictionary with keys 'data', 'indices', 'indptr', 'shape' for CSR construction
    """
    import scipy.sparse as smat

    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert inputs to torch tensors on specified device
    if not isinstance(row_idx, torch.Tensor):
        row_idx = torch.from_numpy(row_idx).to(device)
    else:
        row_idx = row_idx.to(device)

    if not isinstance(col_idx, torch.Tensor):
        col_idx = torch.from_numpy(col_idx).to(device)
    else:
        col_idx = col_idx.to(device)

    if not isinstance(val, torch.Tensor):
        val = torch.from_numpy(val).to(device)
    else:
        val = val.to(device)

    num_rows, num_cols = shape

    # Create COO sparse tensor and convert to CSR (PyTorch handles sorting)
    indices = torch.stack([row_idx, col_idx], dim=0)
    sparse_tensor = torch.sparse_coo_tensor(indices, val, size=shape, device=device)
    sparse_tensor = sparse_tensor.coalesce()  # Combine duplicates

    # Convert to CSR format using PyTorch's internal conversion
    csr_tensor = sparse_tensor.to_sparse_csr()

    # Extract CSR components
    crow_indices = csr_tensor.crow_indices()  # Row pointers (indptr)
    col_indices = csr_tensor.col_indices()    # Column indices
    values = csr_tensor.values()               # Data values

    # Sort each row by value in descending order
    result_rows = []
    result_cols = []
    result_vals = []

    for i in range(num_rows):
        start = crow_indices[i].item()
        end = crow_indices[i + 1].item()

        if start == end:  # Empty row
            continue

        row_values = values[start:end]
        row_cols = col_indices[start:end]

        # Sort in descending order
        if only_topk is not None and len(row_values) > only_topk:
            # Use topk for efficiency when only_topk is specified
            topk_vals, topk_indices = torch.topk(row_values, k=min(only_topk, len(row_values)), largest=True, sorted=True)
            row_values = topk_vals
            row_cols = row_cols[topk_indices]
        else:
            # Full sort
            sorted_indices = torch.argsort(row_values, descending=True)
            row_values = row_values[sorted_indices]
            row_cols = row_cols[sorted_indices]

        # Append to result
        result_vals.append(row_values)
        result_cols.append(row_cols)
        result_rows.extend([i] * len(row_values))

    # Concatenate all results
    if len(result_vals) == 0:
        # Empty matrix
        return {
            'data': np.array([], dtype=np.float32),
            'indices': np.array([], dtype=np.int32),
            'indptr': np.zeros(num_rows + 1, dtype=np.int32),
            'shape': shape
        }

    result_vals = torch.cat(result_vals)
    result_cols = torch.cat(result_cols)
    result_rows = torch.tensor(result_rows, dtype=torch.int32, device=device)

    # Build indptr
    indptr = torch.zeros(num_rows + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.bincount(result_rows + 1, minlength=num_rows)
    indptr = torch.cumsum(indptr, dim=0)

    # Convert back to numpy for CSR construction
    return {
        'data': result_vals.cpu().numpy(),
        'indices': result_cols.cpu().numpy(),
        'indptr': indptr.cpu().numpy(),
        'shape': shape
    }


def torch_topk_per_row(tensor, k, largest=True, sorted=True):
    """Efficient top-k selection per row on GPU.

    Optimized for large batch processing (10M+ rows).

    Args:
        tensor (torch.Tensor): Input tensor of shape (num_rows, num_cols)
        k (int): Number of top elements to select per row
        largest (bool): If True, select largest elements. Default True
        sorted (bool): If True, return sorted results. Default True

    Returns:
        values (torch.Tensor): Top-k values, shape (num_rows, k)
        indices (torch.Tensor): Indices of top-k values, shape (num_rows, k)
    """
    if k >= tensor.shape[1]:
        # If k >= num_cols, just sort all columns
        values, indices = torch.sort(tensor, dim=1, descending=largest)
        return values, indices

    return torch.topk(tensor, k=k, dim=1, largest=largest, sorted=sorted)


def batch_normalize(tensor, dim=1, eps=1e-12):
    """GPU-accelerated L2 normalization.

    Replacement for sklearn.preprocessing.normalize that works on GPU.

    Args:
        tensor (torch.Tensor): Input tensor
        dim (int): Dimension along which to normalize. Default 1 (row-wise)
        eps (float): Small value to avoid division by zero. Default 1e-12

    Returns:
        torch.Tensor: Normalized tensor
    """
    import torch.nn.functional as F
    return F.normalize(tensor, p=2, dim=dim, eps=eps)


def tensor_to_csr(tensor, topk=None):
    """Convert dense PyTorch tensor to SciPy CSR matrix.

    Memory-efficient conversion with optional top-k per row.

    Args:
        tensor (torch.Tensor): Dense tensor of shape (num_rows, num_cols)
        topk (int, optional): Keep only top-k elements per row. Default None

    Returns:
        scipy.sparse.csr_matrix: CSR matrix
    """
    import scipy.sparse as smat

    # Move to CPU if on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert to numpy
    dense_np = tensor.numpy()

    if topk is not None:
        # Use existing smat_util function
        from pecos.utils import smat_util
        return smat_util.dense_to_csr(dense_np, topk=topk)
    else:
        # Direct conversion
        return smat.csr_matrix(dense_np)


def csr_to_tensor(csr_matrix, device=None):
    """Convert SciPy CSR matrix to PyTorch sparse tensor.

    Args:
        csr_matrix (scipy.sparse.csr_matrix): Input CSR matrix
        device (torch.device, optional): Target device. Default None (CPU)

    Returns:
        torch.Tensor: Dense tensor (for compatibility, can be optimized to sparse later)
    """
    if device is None:
        device = torch.device("cpu")

    # Convert to dense for now (can optimize to sparse tensor later)
    dense = torch.from_numpy(csr_matrix.toarray()).to(device)
    return dense


def get_gpu_memory_info():
    """Get current GPU memory usage information.

    Returns:
        dict: Dictionary with 'allocated', 'reserved', 'free' memory in GB
    """
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'free': 0}

    allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - allocated

    return {
        'allocated': allocated,
        'reserved': reserved,
        'free': free,
        'total': total
    }


def auto_batch_size(dataset_size, base_batch_size=32, max_memory_gb=None):
    """Automatically determine optimal batch size based on GPU memory.

    For large datasets (10M+ rows), this helps prevent OOM errors.

    Args:
        dataset_size (int): Total number of samples
        base_batch_size (int): Starting batch size. Default 32
        max_memory_gb (float, optional): Maximum GPU memory to use in GB.
                                          Default None (auto-detect)

    Returns:
        int: Recommended batch size
    """
    if not torch.cuda.is_available():
        return base_batch_size

    mem_info = get_gpu_memory_info()
    available_memory = mem_info['free']

    if max_memory_gb is not None:
        available_memory = min(available_memory, max_memory_gb)

    # Heuristic: Larger batch size for more available memory
    if available_memory > 16:
        multiplier = 4
    elif available_memory > 8:
        multiplier = 2
    else:
        multiplier = 1

    recommended_batch_size = base_batch_size * multiplier

    # Cap at reasonable limits
    recommended_batch_size = min(recommended_batch_size, 512)
    recommended_batch_size = max(recommended_batch_size, 8)

    LOGGER.info(f"Auto batch size: {recommended_batch_size} (Available GPU memory: {available_memory:.2f} GB)")

    return recommended_batch_size
