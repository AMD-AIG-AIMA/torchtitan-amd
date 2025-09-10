import triton
import triton.language as tl


@triton.jit
def get_flat_tid():
    """Get flattened thread ID within block"""
    return tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0) + tl.program_id(1) * tl.num_programs(0) + tl.program_id(0)


@triton.jit  
def get_flat_bid():
    """Get flattened block ID"""
    return tl.program_id(0)


@triton.jit
def sync_threads():
    """Thread synchronization - uses Triton's built-in barrier"""
    tl.debug_barrier()