
import torch
import torch.nn as nn
import torch.distributed as dist



# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage
_PIPELINE_GLOBAL_RANKS = None


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    #assert _DATA_PARALLEL_GROUP is not None, f"data parallel group={_DATA_PARALLEL_GROUP}, which is not initialized"
    return _DATA_PARALLEL_GROUP


def get_group_world_size_rank():
    group = get_data_parallel_group()
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)
    return group, rank, world_size


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


class AllgatherFromDataParallelRegion(torch.autograd.Function):
    """ https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_ict.py#L57 """
    @staticmethod
    def forward(ctx, input_):
        assert input_.dim() == 2
        group, rank, world_size = get_group_world_size_rank()
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        torch.distributed.all_gather(tensor_list, input_, group=group)
        tensor_list[rank] = input_
        output = torch.cat(tensor_list, dim=0).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        group, rank, world_size = get_group_world_size_rank()
        assert grad_output.shape[0] % world_size == 0
        dim_size = grad_output.shape[0] // world_size
        output_list = torch.split(grad_output, dim_size, dim=0)
        # get chunk from this rank
        output = output_list[rank].contiguous()
        return output


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=get_data_parallel_group())
    averaged_losses = averaged_losses / torch.distributed.get_world_size(group=get_data_parallel_group())
    return averaged_losses
