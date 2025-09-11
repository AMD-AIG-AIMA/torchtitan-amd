from typing import Optional
import torch
import torch.nn as nn
import torch.distributed as dist
from torchtitan.models.moe import MoEArgs
from torchtitan.tools.logging import logger
from dataclasses import dataclass


def get_expert_parallel_group(dim_name: str = "dp_shard_in_ep"):
    """Get expert parallel process group if available"""
    # try:
    #     import torch.distributed.device_mesh as device_mesh
    #     current_mesh = device_mesh._mesh_resources.get_current_mesh()
    #     if current_mesh is None:
    #         return None
    #     return current_mesh.get_group(dim_name)
    # except (ImportError, AttributeError, RuntimeError):
    #     return None
    import torch.distributed.device_mesh as device_mesh
    current_mesh = device_mesh._mesh_resources.get_current_mesh()
    if current_mesh is None:
        return None
    return current_mesh.get_group(dim_name)


@dataclass
class DeepSeekConfig:
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 16
    n_shared_experts: Optional[int] = None
    num_experts_per_tok: int = 2
    max_seq_len: int = 4096
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = False
    routed_scaling_factor: float = 1.0
    ep_size: int = 1


def create_deepseek_config(moe_args: MoEArgs, dim: int, hidden_dim: int, max_seq_len: int):
    """Convert LLaMA4 MoEArgs to DeepSeek V3 compatible config"""
    # try:
    #     from torchtitan.experiments.deepseek_v3.model_config import ModelArgs as DeepSeekConfig
    # except ImportError:
    #     # Create minimal config if DeepSeek not available
    #     from dataclasses import dataclass
        
    #     @dataclass
    #     class DeepSeekConfig:
    #         hidden_size: int = 4096
    #         moe_intermediate_size: int = 2048
    #         n_routed_experts: int = 8
    #         n_shared_experts: Optional[int] = None
    #         num_experts_per_tok: int = 2
    #         max_seq_len: int = 4096
    #         scoring_func: str = "sigmoid"
    #         norm_topk_prob: bool = False
    #         routed_scaling_factor: float = 1.0
    #         ep_size: int = 1
    
    config = DeepSeekConfig()
    
    # Map LLaMA4 parameters to DeepSeek V3
    config.hidden_size = dim
    config.moe_intermediate_size = hidden_dim
    # config.n_routed_experts = moe_args.num_experts
    config.n_routed_experts = 16
    config.n_shared_experts = moe_args.num_shared_experts if moe_args.num_shared_experts > 0 else None
    config.num_experts_per_tok = moe_args.top_k
    config.max_seq_len = max_seq_len
    config.scoring_func = moe_args.score_func
    config.norm_topk_prob = moe_args.route_norm
    config.routed_scaling_factor = moe_args.route_scale
    
    return config


class LLaMA4SymmMemMoE(nn.Module):
    """
    Hybrid MoE that combines LLaMA4 interface with DeepSeek V3 symmetric memory.
    
    Automatically chooses the appropriate implementation:
    - Standard MoE: Single node or when expert parallelism is disabled
    - DeepSeek V3 MoE: Multi-node with expert parallelism and symmetric memory
    
    Args:
        moe_args: LLaMA4 MoE configuration
        dim: Model hidden dimension
        hidden_dim: FFN hidden dimension
        ep_enabled: Whether expert parallelism is enabled
        max_seq_len: Maximum sequence length for buffer allocation
    """

    def __init__(
        self,
        moe_args: MoEArgs,
        dim: int,
        hidden_dim: int,
        ep_enabled: bool = False,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        
        self.moe_args = moe_args
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Determine if we should use distributed MoE
        # use_distributed = self._should_use_distributed_moe(ep_enabled)
        use_distributed = True
        
        if use_distributed:
            self._init_distributed_moe()
        else:
            self._init_standard_moe()
    
    def _should_use_distributed_moe(self, ep_enabled: bool) -> bool:
        """Determine whether to use distributed DeepSeek V3 MoE"""
        if not ep_enabled:
            return False
        
        if not dist.is_initialized():
            logger.info("Distributed training not initialized, using standard MoE")
            return False
        
        ep_group = get_expert_parallel_group()
        if ep_group is None:
            logger.info("Expert parallel group not found, using standard MoE")
            return False
        
        ep_size = ep_group.size()
        if ep_size <= 1:
            logger.info("Expert parallel size <= 1, using standard MoE")
            return False
        
        logger.info(f"Using distributed DeepSeek V3 MoE (ep_size={ep_size})")
        return True
    
    def _init_distributed_moe(self):
        """Initialize DeepSeek V3 MoE with symmetric memory"""
        try:
            from .deepseek_v3_components import DeepSeekV3MoE
            
            config = create_deepseek_config(
                self.moe_args, self.dim, self.hidden_dim, self.max_seq_len
            )
            
            # Set expert parallel size
            ep_group = get_expert_parallel_group()
            config.ep_size = ep_group.size() if ep_group else 1
            
            self.moe_impl = DeepSeekV3MoE(config)
            self.symmetric_memory_enabled = True
            self.implementation_type = "deepseek_v3"
            
        except ImportError as e:
            logger.warning(f"Failed to import DeepSeek V3 components: {e}")
            logger.warning("Falling back to standard MoE")
            self._init_standard_moe()
    
    def _init_standard_moe(self):
        """Initialize standard LLaMA4 MoE"""
        from torchtitan.models.moe import MoE
        
        self.moe_impl = MoE(self.moe_args, self.dim, self.hidden_dim)
        self.symmetric_memory_enabled = False
        self.implementation_type = "standard"
        
        logger.info("Using standard LLaMA4 MoE implementation")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MoE layer"""
        return self.moe_impl(x)
    
    def setup_symmetric_memory(self, dtype: torch.dtype, device: torch.device):
        """Setup symmetric memory buffers if supported"""
        if not self.symmetric_memory_enabled:
            logger.debug("Symmetric memory not enabled")
            return
        
        if not hasattr(self.moe_impl, 'setup_symm_mem'):
            logger.warning("MoE implementation does not support symmetric memory")
            return
        
        try:
            logger.info("Setting up symmetric memory for MoE")
            self.moe_impl.setup_symm_mem(dtype, device)
        except Exception as e:
            logger.warning(f"Failed to setup symmetric memory: {e}")
            logger.warning("MoE will continue with standard all-to-all communication")
    
    def init_weights(self, init_std: float, buffer_device: torch.device):
        """Initialize MoE weights"""
        if hasattr(self.moe_impl, 'init_weights'):
            self.moe_impl.init_weights(init_std, buffer_device)
    
    @property
    def is_symmetric_memory_enabled(self) -> bool:
        """Check if symmetric memory is enabled"""
        return self.symmetric_memory_enabled
    
    @property
    def expert_parallel_size(self) -> int:
        """Get expert parallel size"""
        if hasattr(self.moe_impl, 'ep_size'):
            return self.moe_impl.ep_size
        return 1
    
    @property
    def experts(self):
        """Get experts from the underlying MoE implementation"""
        if hasattr(self.moe_impl, 'experts'):
            return self.moe_impl.experts
        return None
    
    @property
    def load_balance_coeff(self):
        """Get load_balance_coeff from the underlying MoE implementation"""
        if hasattr(self.moe_impl, 'load_balance_coeff'):
            return self.moe_impl.load_balance_coeff
        return None
    
    @property
    def tokens_per_expert(self):
        """Get tokens_per_expert from the underlying MoE implementation"""
        if hasattr(self.moe_impl, 'tokens_per_expert'):
            return self.moe_impl.tokens_per_expert
        return None
    
    @property
    def expert_bias(self):
        """Get expert_bias from the underlying MoE implementation"""
        if hasattr(self.moe_impl, 'expert_bias'):
            return self.moe_impl.expert_bias
        return None
    
    def __repr__(self) -> str:
        return (
            f"LLaMA4SymmMemMoE("
            f"type={self.implementation_type}, "
            f"experts={self.moe_args.num_experts}, "
            f"top_k={self.moe_args.top_k}, "
            f"ep_size={self.expert_parallel_size}, "
            f"symm_mem={'enabled' if self.symmetric_memory_enabled else 'disabled'}"
            f")"
        )