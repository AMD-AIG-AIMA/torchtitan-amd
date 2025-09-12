from typing import Optional
import torch
import torch.nn as nn
import torch.distributed as dist
from torchtitan.models.moe import MoEArgs
from torchtitan.tools.logging import logger
from dataclasses import dataclass

from .deepseek_v3_components import DeepSeekV3MoE, get_expert_parallel_group


@dataclass
class DeepSeekConfig:
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_experts: int = 16
    n_shared_experts: Optional[int] = None
    top_k: int = 1
    score_func: str = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    ep_size: int = 1


def create_deepseek_config(moe_args: MoEArgs, dim: int, hidden_dim: int):
    """Convert LLaMA4 MoEArgs to DeepSeek V3 compatible config"""
    config = DeepSeekConfig()
    
    # Map LLaMA4 parameters to DeepSeek V3
    config.hidden_size = dim
    config.moe_intermediate_size = hidden_dim
    config.num_experts = moe_args.num_experts
    config.n_shared_experts = moe_args.num_shared_experts if moe_args.num_shared_experts > 0 else None
    config.top_k = moe_args.top_k
    config.score_func = moe_args.score_func
    config.route_norm = moe_args.route_norm
    config.route_scale = moe_args.route_scale
    
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
    """

    def __init__(
        self,
        moe_args: MoEArgs,
        dim: int,
        hidden_dim: int,
        ep_enabled: bool = False,
    ):
        super().__init__()

        self.moe_args = moe_args
        self.dim = dim
        self.hidden_dim = hidden_dim
            
        deepseek_config = create_deepseek_config(self.moe_args, self.dim, self.hidden_dim)
            
        ep_group = get_expert_parallel_group()
        deepseek_config.ep_size = ep_group.size() if ep_group else 1
            
        self.moe_impl = DeepSeekV3MoE(deepseek_config)
        self.symmetric_memory_enabled = True
        self.implementation_type = "deepseek_v3"
        
        # logger.warning(f"Failed to import DeepSeek V3 components: {e}")
        # logger.warning("Falling back to standard MoE")
        # self._init_standard_moe()
    
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
        logger.info("Setting up symmetric memory for MoE")
        logger.info("Implementation type: %s %s" % (self.implementation_type, type(self.moe_impl)))
        self.moe_impl.setup_symm_mem(dtype, device)
    
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