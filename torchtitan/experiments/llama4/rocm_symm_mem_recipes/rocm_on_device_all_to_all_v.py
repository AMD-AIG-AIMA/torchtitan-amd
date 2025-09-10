import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import is_hip

from .rocm_barrier import blockwise_barrier
from .rocm_utils import sync_threads


@triton.jit
def _rocm_exchange_row_offsets(
    split_sizes_ptrs,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
):
    """
    ROCm-compatible row offset exchange computation.
    Uses same logic as original but with ROCm-optimized memory operations.
    """
    remote_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK

    # split_sizes_ptr for all ranks
    split_sizes_ptrs = split_sizes_ptrs.to(tl.pointer_type(tl.uint64))

    # split_sizes_matrix[remote_rank, :]
    input_split_sizes_ptr = tl.load(split_sizes_ptrs + remote_rank).to(
        tl.pointer_type(tl.int64)
    )

    offsets_ = tl.arange(0, world_size)
    input_split_sizes = tl.load(
        input_split_sizes_ptr + offsets_, mask=offsets_ <= rank, other=0
    )

    num_rows = tl.load(input_split_sizes_ptr + rank)
    input_row_offset = tl.sum(input_split_sizes) - num_rows

    # split_sizes_matrix[:, rank]
    output_split_sizes_ptrs = (
        tl.load(split_sizes_ptrs + offsets_).to(tl.pointer_type(tl.int64)) + rank
    )
    output_split_sizes = tl.load(
        output_split_sizes_ptrs, mask=offsets_ <= remote_rank, other=0
    )
    output_row_offset = tl.sum(output_split_sizes) - num_rows

    return input_row_offset, output_row_offset, num_rows


@triton.jit
def rocm_on_device_all_to_all_v_kernel(
    output_ptr,
    output_splits_ptr,
    input_ptrs,
    input_splits_ptr,
    signal_pad_ptrs,
    dim: tl.constexpr,  # Separate dim for easier vectorization
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
    UNROLL_FACTOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    ROCm-compatible kernel for on-device all-to-all operations.
    
    Replaces PTX-based barriers with ROCm-compatible generic atomic operations
    while maintaining the same high-performance data movement patterns.
    """
    # Pre-synchronization barrier using ROCm-compatible atomics
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    sync_threads()

    remote_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK
    block_offset = tl.program_id(0) % BLOCKS_PER_REMOTE_RANK

    # Calculate row offsets for data exchange
    input_row_offset, output_row_offset, num_rows = _rocm_exchange_row_offsets(
        input_splits_ptr, rank, world_size, BLOCKS_PER_REMOTE_RANK
    )

    # Update output splits (only first block per remote rank)
    output_splits_ptr = output_splits_ptr.to(tl.pointer_type(tl.uint64))
    if block_offset == 0:
        tl.store(output_splits_ptr + remote_rank, num_rows)

    # Calculate input and output pointers
    input_ptr = (
        tl.load(input_ptrs.to(tl.pointer_type(tl.uint64)) + remote_rank).to(
            tl.pointer_type(tl.bfloat16)
        )
        + input_row_offset * dim
    )
    output_ptr = output_ptr + output_row_offset * dim

    # Optimized data copying with unrolling
    outer_loop_step = BLOCK_SIZE * UNROLL_FACTOR
    outer_loop_iters_per_rank = tl.cdiv(
        tl.cdiv(num_rows * dim, outer_loop_step), BLOCKS_PER_REMOTE_RANK
    )
    numel_per_rank = outer_loop_step * outer_loop_iters_per_rank
    offset = numel_per_rank * block_offset
    end = tl.minimum(numel_per_rank * (block_offset + 1), num_rows * dim)

    # Unrolled copying for better performance
    unroll_region_size = (end - offset) // outer_loop_step * outer_loop_step
    for i in tl.range(offset, offset + unroll_region_size, outer_loop_step):
        for j in tl.range(
            i,
            i + outer_loop_step,
            BLOCK_SIZE,
            loop_unroll_factor=UNROLL_FACTOR,
        ):
            offsets = j + tl.arange(0, BLOCK_SIZE)
            data = tl.load(input_ptr + offsets)
            tl.store(output_ptr + offsets, data)

    # Handle remaining elements
    offset += unroll_region_size
    while offset < end:
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_rows * dim
        data = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, data, mask=mask)
        offset += BLOCK_SIZE

    # Post-synchronization barrier using ROCm-compatible atomics
    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")


def _rocm_on_device_all_to_all_v(
    output: torch.Tensor,
    output_splits: torch.Tensor,
    input: torch.Tensor,
    input_splits: torch.Tensor,
    group: dist.ProcessGroup = dist.group.WORLD,
    BLOCKS_PER_REMOTE_RANK=8,
    UNROLL_FACTOR: int = 8,
    BLOCK_SIZE: int = 16384,
):
    """
    ROCm-optimized implementation of on-device all-to-all.
    
    Uses ROCm-compatible barriers and optimized kernel parameters for AMD GPUs.
    Falls back to standard distributed all-to-all if ROCm optimizations fail.
    """
    assert output.dim() == 2, f"Expected 2D output tensor, got {output.shape}"
    assert input.dim() == 2, f"Expected 2D input tensor, got {input.shape}"
    assert output.shape[1] == input.shape[1], f"Dimension mismatch: {output.shape[1]} != {input.shape[1]}"

    dim = output.shape[1]
    
    try:
        # Set up symmetric memory handles
        input_hdl = symm_mem.rendezvous(input, group=group)
        input_splits_hdl = symm_mem.rendezvous(input_splits, group=group)

        # Launch ROCm-optimized kernel
        num_blocks = input_hdl.world_size * BLOCKS_PER_REMOTE_RANK
        
        # Adjust kernel parameters for ROCm
        num_warps = 16 if is_hip() else 16  # Could be tuned for specific AMD GPUs
        
        kernel = rocm_on_device_all_to_all_v_kernel[(num_blocks, 1, 1)](
            output,
            output_splits,
            input_hdl.buffer_ptrs_dev,
            input_splits_hdl.buffer_ptrs_dev,
            input_hdl.signal_pad_ptrs_dev,
            dim=dim,
            rank=input_hdl.rank,
            world_size=input_hdl.world_size,
            BLOCKS_PER_REMOTE_RANK=BLOCKS_PER_REMOTE_RANK,
            UNROLL_FACTOR=UNROLL_FACTOR,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        logger.debug("ROCm on-device all-to-all kernel completed successfully")
        return output
        
    except Exception as e:
        logger.warning(f"ROCm on-device all-to-all failed: {e}")
        logger.warning("Falling back to standard distributed all-to-all")
        
        # Fallback to standard distributed all-to-all
        return _fallback_all_to_all_v(output, output_splits, input, input_splits, group)


def _fallback_all_to_all_v(
    output: torch.Tensor,
    output_splits: torch.Tensor, 
    input: torch.Tensor,
    input_splits: torch.Tensor,
    group: dist.ProcessGroup,
):
    """
    Fallback implementation using standard PyTorch distributed all-to-all.
    
    This provides functional correctness when ROCm-optimized path fails.
    """
    try:
        # Convert to list format for torch.distributed.all_to_all_single
        input_list = []
        output_list = []
        
        input_offset = 0
        output_offset = 0
        
        for i, (in_size, out_size) in enumerate(zip(input_splits.tolist(), output_splits.tolist())):
            if in_size > 0:
                input_list.append(input[input_offset:input_offset + in_size])
                input_offset += in_size
            else:
                input_list.append(torch.empty(0, input.shape[1], dtype=input.dtype, device=input.device))
                
            if out_size > 0:
                output_list.append(output[output_offset:output_offset + out_size])
                output_offset += out_size
            else:
                output_list.append(torch.empty(0, output.shape[1], dtype=output.dtype, device=output.device))
        
        # Perform standard all-to-all
        dist.all_to_all(output_list, input_list, group=group)
        
        # Copy results back to output tensor
        output_offset = 0
        for i, out_tensor in enumerate(output_list):
            if out_tensor.numel() > 0:
                output[output_offset:output_offset + out_tensor.shape[0]] = out_tensor
                output_offset += out_tensor.shape[0]
        
        logger.debug("Fallback all-to-all completed successfully")
        return output
        
    except Exception as e:
        logger.error(f"Both ROCm and fallback all-to-all failed: {e}")
        raise RuntimeError("All all-to-all implementations failed") from e


class ROCmOnDeviceAllToAllV(torch.autograd.Function):
    """
    ROCm-compatible version of OnDeviceAllToAllV using generic atomic operations.
    
    This class maintains the same interface as the original but uses ROCm-compatible
    implementations that work on AMD GPUs without PTX inline assembly.
    """
    # Symmetric memory buffers (shared across instances)
    grad_output_buf = None
    splits_buf = None
    max_output_len = None

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        input_splits: torch.Tensor,
        group: dist.ProcessGroup = dist.group.WORLD,
    ):
        """
        Forward pass with ROCm-compatible symmetric memory operations.
        
        Args:
            input: Input tensor with data for all ranks concatenated
            input_splits: Input splits of shape (group.world_size,)
            group: Process group to scope the collective
        """
        # Initialize input splits buffer (one time only)
        if ROCmOnDeviceAllToAllV.splits_buf is None:
            ROCmOnDeviceAllToAllV.splits_buf = symm_mem.empty(
                *input_splits.shape,
                dtype=input_splits.dtype,
                device=input_splits.device,
            )

        if ROCmOnDeviceAllToAllV.max_output_len is None:
            raise RuntimeError(
                "Please set max output length via `ROCmOnDeviceAllToAllV.max_output_len = ...`"
            )

        # Allocate output buffers
        output = input.new_empty(ROCmOnDeviceAllToAllV.max_output_len, *input.shape[1:])
        output_splits = torch.empty_like(input_splits)
        
        # Copy input splits to symmetric memory buffer
        ROCmOnDeviceAllToAllV.splits_buf.copy_(input_splits)

        # Perform ROCm-optimized shuffle
        _rocm_on_device_all_to_all_v(
            output, output_splits, input, ROCmOnDeviceAllToAllV.splits_buf, group=group
        )

        # Save context for backward pass
        ctx.save_for_backward(output_splits)
        ctx.group = group
        ctx.input_shape = input.shape
        
        return output, output_splits

    @staticmethod
    def backward(ctx, grad_output, grad_splits):
        """
        Backward pass with ROCm-compatible operations.
        
        Implements backward as a shuffle of output gradients to input using
        the same ROCm-optimized kernels.
        """
        # Initialize grad_output buffer (one time only)
        if ROCmOnDeviceAllToAllV.grad_output_buf is None:
            assert (
                ROCmOnDeviceAllToAllV.max_output_len is not None
            ), "`max_output_len` not set"
            ROCmOnDeviceAllToAllV.grad_output_buf = symm_mem.empty(
                ROCmOnDeviceAllToAllV.max_output_len,
                *grad_output.shape[1:],
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

        # Copy gradients to symmetric memory buffer
        ROCmOnDeviceAllToAllV.grad_output_buf.narrow(0, 0, grad_output.shape[0]).copy_(
            grad_output
        )

        # Retrieve saved context
        (grad_output_splits,) = ctx.saved_tensors
        ROCmOnDeviceAllToAllV.splits_buf.copy_(grad_output_splits)
        grad_input_splits = torch.empty_like(grad_output_splits)  # unused
        grad_input = grad_output.new_empty(*ctx.input_shape)

        # Shuffle gradients back using ROCm-optimized operations
        _rocm_on_device_all_to_all_v(
            grad_input,
            grad_input_splits,
            ROCmOnDeviceAllToAllV.grad_output_buf,
            ROCmOnDeviceAllToAllV.splits_buf,
            group=ctx.group,
        )
        
        return grad_input, None, None


# Convenience alias
rocm_on_device_all_to_all_v = ROCmOnDeviceAllToAllV.apply


def is_rocm_symmetric_memory_available():
    """Check if ROCm symmetric memory operations are available"""
    try:
        return (
            is_hip() and
            torch.distributed.is_available() and
            hasattr(torch.distributed, '_symmetric_memory')
        )
    except Exception:
        return False