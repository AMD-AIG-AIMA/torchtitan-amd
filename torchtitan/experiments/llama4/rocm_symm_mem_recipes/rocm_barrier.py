import triton
import triton.language as tl
from torchtitan.tools.logging import logger

from .rocm_utils import get_flat_bid, get_flat_tid


@triton.jit
def rocm_send_signal(addrs, sem: tl.constexpr):
    """
    ROCm-compatible signal sending using generic atomic operations.
    
    Replaces PTX inline assembly with portable Triton atomic operations
    that work on both NVIDIA and AMD GPUs.
    
    Args:
        addrs: Memory addresses to signal
        sem: Semantic ordering ("relaxed" or "acq_rel")
    """
    if sem == "relaxed":
        # Use generic Triton atomic compare-and-swap instead of PTX
        # This spins until we successfully change 0 -> 1 (acquire the signal)
        old_val = tl.atomic_cas(addrs, 0, 1)
        # If old_val was not 0, someone else got it first, so we spin
        while old_val != 0:
            old_val = tl.atomic_cas(addrs, 0, 1)
            
    elif sem == "acq_rel":
        # For acquire-release semantics, we use the same CAS but with 
        # implicit memory ordering that Triton provides
        old_val = tl.atomic_cas(addrs, 0, 1) 
        while old_val != 0:
            old_val = tl.atomic_cas(addrs, 0, 1)
    else:
        # Compile-time error for unsupported semantics
        tl.static_assert(False, "Unrecognized sem, must be 'relaxed' or 'acq_rel'")


@triton.jit
def rocm_wait_signal(addrs, sem: tl.constexpr):
    """
    ROCm-compatible signal waiting using generic atomic operations.
    
    Replaces PTX inline assembly with portable Triton atomic operations
    that work on both NVIDIA and AMD GPUs.
    
    Args:
        addrs: Memory addresses to wait on
        sem: Semantic ordering ("relaxed" or "acq_rel")
    """
    if sem == "relaxed":
        # Wait for signal to be set (1), then clear it back to 0
        old_val = tl.atomic_cas(addrs, 1, 0)
        while old_val != 1:
            old_val = tl.atomic_cas(addrs, 1, 0)
            
    elif sem == "acq_rel":
        # Same logic but with acquire-release semantics
        old_val = tl.atomic_cas(addrs, 1, 0)
        while old_val != 1:
            old_val = tl.atomic_cas(addrs, 1, 0)
    else:
        # Compile-time error for unsupported semantics  
        tl.static_assert(False, "Unrecognized sem, must be 'relaxed' or 'acq_rel'")


@triton.jit
def blockwise_barrier(
    signal_pad_ptrs,
    block_id,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    sem: tl.constexpr,
):
    """
    ROCm-compatible blockwise barrier for symmetric memory synchronization.
    
    This function synchronizes blocks with matching block_id across participating devices
    using generic Triton atomic operations instead of PTX inline assembly.
    
    Synchronization patterns:
    - Pattern 0 (relaxed): Ensures all previous writes are visible
    - Pattern 1 (acq_rel): Ensures current writes are visible to remote blocks  
    - Pattern 2 (relaxed): Ensures buffers are safe for subsequent writes
    
    Args:
        signal_pad_ptrs: Pointers to signal pad buffers for each rank
        block_id: Block identifier for matching across devices (None = auto-detect)
        rank: Current device rank
        world_size: Total number of participating devices
        sem: Memory ordering semantics ("relaxed" or "acq_rel")
    """
    # Auto-detect block ID if not provided
    if block_id is None:
        block_id = get_flat_bid()
    flat_tid = get_flat_tid()

    # Calculate signal addresses for sending (to remote ranks)
    remote_ranks = tl.arange(0, world_size)
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks).to(
        tl.pointer_type(tl.uint32)
    )
    send_addrs = remote_signal_pad_addrs + block_id * world_size + rank

    # Calculate signal addresses for waiting (from remote ranks)
    local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(
        tl.pointer_type(tl.uint32)
    )
    wait_addrs = local_signal_pad_addr + block_id * world_size + remote_ranks

    # Only first 'world_size' threads participate in signaling
    if flat_tid < world_size:
        # Send signal to all remote ranks (including self)
        rocm_send_signal(send_addrs, sem)
        # Wait for signals from all remote ranks (including self)  
        rocm_wait_signal(wait_addrs, sem)


def check_rocm_barrier_availability():
    """Check if ROCm barrier operations are available"""
    try:
        from torchtitan.tools.utils import is_hip
        return is_hip()
    except ImportError:
        logger.warning("Could not check HIP availability, assuming CUDA")
        return False