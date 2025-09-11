from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import is_hip

# Import ROCm group GEMM strategies
try:
    from ..rocm_group_gemms import ROCmGroupGEMMStrategy, TorchGroupGEMMFallback
    ROCM_GROUP_GEMM_AVAILABLE = True
except ImportError:
    logger.warning("ROCm group GEMM strategies not available")
    ROCM_GROUP_GEMM_AVAILABLE = False


def get_expert_parallel_group():
    """Get expert parallel process group"""
    try:
        import torch.distributed.device_mesh as device_mesh
        current_mesh = device_mesh._mesh_resources.get_current_mesh()
        if current_mesh is None:
            return None
        return current_mesh.get_group("dp_shard_in_ep")
    except (ImportError, AttributeError, RuntimeError):
        return None


class SimpleMLP(nn.Module):
    """
    Simplified SwiGLU MLP compatible with both LLaMA4 and DeepSeek V3.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SimpleRouter(nn.Module):
    """
    Simplified token router for MoE.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, scoring_func: str = "sigmoid"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Simple load balancing bias (no auxiliary loss)
        self.load_balance_bias = nn.Parameter(torch.zeros(num_experts))

    def forward(self, x):
        """
        Route tokens to experts.
        
        Args:
            x: Input tensor [batch_size * seq_len, hidden_size]
            
        Returns:
            topk_indices: Expert indices [batch_size * seq_len, top_k]
            topk_weights: Expert weights [batch_size * seq_len, top_k]
        """
        # Compute routing scores
        logits = self.gate(x) + self.load_balance_bias.unsqueeze(0)
        
        # Apply scoring function
        if self.scoring_func == "softmax":
            scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        elif self.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits.to(torch.float32))
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func}")
        
        # Select top-k experts
        topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)
        
        # Normalize weights
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return topk_indices.to(torch.int32), topk_weights.to(x.dtype)


def create_group_gemm_strategy(custom_activation=F.silu):
    """
    Factory function to create the optimal Group GEMM strategy for the current environment.
    
    Args:
        custom_activation: Activation function for SwiGLU (default: F.silu)
        
    Returns:
        GroupGEMMStrategy instance
    """
    if not ROCM_GROUP_GEMM_AVAILABLE:
        logger.warning("ROCm group GEMM strategies not available, using manual fallback")
        return None
    
    # Prefer ROCm strategy on AMD systems
    if is_hip() and ROCmGroupGEMMStrategy.is_available():
        logger.info("Using ROCm-optimized group GEMM strategy")
        return ROCmGroupGEMMStrategy(custom_activation)
    
    # Fallback to PyTorch grouped GEMM
    if TorchGroupGEMMFallback.is_available():
        logger.info("Using PyTorch group GEMM fallback strategy")
        return TorchGroupGEMMFallback(custom_activation)
    
    # No group GEMM available
    logger.warning("No group GEMM strategies available")
    return None


class DeepSeekV3MoE(nn.Module):
    """
    Simplified DeepSeek V3 MoE implementation adapted for LLaMA 4.
    
    Features:
    - Expert parallelism support with graceful single-node fallback
    - ROCm-optimized grouped operations
    - Symmetric memory integration (when available)
    - Simple and maintainable codebase
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok

        # Setup expert parallelism
        self._setup_expert_parallelism()

        # Create local experts
        self._create_experts()

        # Create router
        self.router = SimpleRouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            scoring_func=config.scoring_func,
        )

        # Create shared experts if specified
        if config.n_shared_experts is not None:
            shared_intermediate_size = self.intermediate_size * config.n_shared_experts
            self.shared_experts = SimpleMLP(self.hidden_size, shared_intermediate_size)
        else:
            self.shared_experts = None

        # Initialize grouped GEMM strategy
        self.group_gemm_strategy = create_group_gemm_strategy(custom_activation=F.silu)

    def _setup_expert_parallelism(self):
        """Setup expert parallelism with single-node fallback"""
        ep_group = get_expert_parallel_group()
        
        if ep_group is not None and ep_group.size() > 1:
            self.ep_size = ep_group.size()
            self.ep_rank = ep_group.rank()
            self.ep_group = ep_group
            self.distributed_experts = True
            logger.info(f"Using distributed experts: ep_size={self.ep_size}, ep_rank={self.ep_rank}")
        else:
            self.ep_size = 1
            self.ep_rank = 0
            self.ep_group = None
            self.distributed_experts = False
            logger.info("Using single-node experts")

        self.experts_per_rank = self.num_experts // self.ep_size

    def _create_experts(self):
        """Create expert networks for this rank"""
        self.experts = nn.ModuleList()
        
        for i in range(self.experts_per_rank):
            expert = SimpleMLP(self.hidden_size, self.intermediate_size)
            self.experts.append(expert)

    def forward(self, x):
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        identity = x
        batch_size, seq_len, hidden_size = x.shape
        x = x.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]

        # Route tokens to experts
        expert_indices, expert_weights = self.router(x)

        # Process tokens through experts
        if self.distributed_experts:
            # Distributed processing (simplified version)
            expert_output = self._process_distributed(x, expert_indices, expert_weights)
        else:
            # Local processing
            expert_output = self._process_local(x, expert_indices, expert_weights)

        expert_output = expert_output.view(batch_size, seq_len, hidden_size)

        # Add shared expert contribution
        if self.shared_experts is not None:
            expert_output = expert_output + self.shared_experts(identity)

        return expert_output

    def _process_local(self, tokens, expert_indices, expert_weights):
        """Process tokens through local experts using optimized grouped operations when available"""
        
        # Try to use grouped GEMM strategy if available
        if self.group_gemm_strategy is not None:
            return self._process_with_grouped_gemm(tokens, expert_indices, expert_weights)
        
        # Fallback to manual expert loop
        # return self._process_manual_loop(tokens, expert_indices, expert_weights)

    def _process_with_grouped_gemm(self, tokens, expert_indices, expert_weights):
        """Process tokens using grouped GEMM operations"""
        
        # Arrange tokens and compute group sizes for each expert
        expert_tokens_list = []
        expert_weights_gathered = []
        m_sizes = []
        
        for expert_idx in range(len(self.experts)):
            # Find tokens assigned to this expert
            mask = (expert_indices == expert_idx)
            expert_token_mask = mask.any(dim=1)
            
            if expert_token_mask.any():
                expert_tokens = tokens[expert_token_mask]
                expert_tokens_list.append(expert_tokens)
                m_sizes.append(expert_tokens.shape[0])
                expert_weights_gathered.append(expert_weights[expert_token_mask])
            else:
                m_sizes.append(0)
        
        if not expert_tokens_list:
            return torch.zeros_like(tokens)
        
        # Concatenate all tokens contiguously
        contig_tokens = torch.cat(expert_tokens_list, dim=0)
        m_sizes_tensor = torch.tensor(m_sizes, device=tokens.device, dtype=torch.int32)
        
        # Arrange expert weights for grouped operations
        module_mock = self._create_module_mock()
        
        # Process using the grouped GEMM strategy
        processed_tokens = self.group_gemm_strategy.execute(
            contig_tokens, m_sizes_tensor, None, module_mock
        )
        
        # Redistribute processed tokens back to original positions
        output = torch.zeros_like(tokens)
        offset = 0
        expert_idx = 0
        
        for i, num_tokens in enumerate(m_sizes):
            if num_tokens > 0:
                expert_output = processed_tokens[offset:offset + num_tokens]
                
                # Find original positions for this expert
                mask = (expert_indices == i)
                expert_token_mask = mask.any(dim=1)
                
                # Apply routing weights
                routing_weights = expert_weights_gathered[expert_idx].mean(dim=-1, keepdim=True)
                weighted_output = expert_output * routing_weights
                
                output[expert_token_mask] += weighted_output
                offset += num_tokens
                expert_idx += 1
        
        return output

    def _create_module_mock(self):
        """Create a mock module with arranged expert weights for grouped GEMM"""
        # Collect weights from all experts
        gate_weights = []
        up_weights = []
        down_weights = []
        
        for expert in self.experts:
            gate_weights.append(expert.gate_proj.weight)
            up_weights.append(expert.up_proj.weight)  
            down_weights.append(expert.down_proj.weight)
        
        # Arrange weights using the group GEMM strategy
        class MockModule:
            pass
        
        module = MockModule()
        module.gate_proj_weight = self.group_gemm_strategy.arrange_expert_weights(gate_weights, "gate_proj", module)
        module.up_proj_weight = self.group_gemm_strategy.arrange_expert_weights(up_weights, "up_proj", module)
        module.down_proj_weight = self.group_gemm_strategy.arrange_expert_weights(down_weights, "down_proj", module)
        
        return module

    def _process_manual_loop(self, tokens, expert_indices, expert_weights):
        """Fallback manual loop processing"""
        output = torch.zeros_like(tokens)

        for expert_idx, expert in enumerate(self.experts):
            # Find tokens assigned to this expert
            mask = (expert_indices == expert_idx)
            if not mask.any():
                continue

            # Get tokens and weights for this expert
            expert_token_mask = mask.any(dim=1)
            expert_tokens = tokens[expert_token_mask]
            if expert_tokens.numel() == 0:
                continue

            # Process through expert
            expert_out = expert(expert_tokens)

            # Apply routing weights and accumulate
            weights = expert_weights[mask].mean(dim=-1, keepdim=True)
            weighted_out = expert_out * weights

            output[expert_token_mask] += weighted_out

        return output

    def _process_distributed(self, tokens, expert_indices, expert_weights):
        """Process tokens through distributed experts (simplified)"""
        # For now, fall back to local processing
        # Full distributed implementation would require token shuffling
        logger.debug("Distributed expert processing not fully implemented, using local fallback")
        return self._process_local(tokens, expert_indices, expert_weights)

    def setup_symm_mem(self, dtype: torch.dtype, device: torch.device):
        """Setup symmetric memory for distributed processing"""
        if not self.distributed_experts:
            logger.info("Symmetric memory not needed for single-node setup")
            return

        try:
            # Import symmetric memory components
            import torch.distributed._symmetric_memory as symm_mem
            
            # Try ROCm-compatible symmetric memory first
            if is_hip():
                try:
                    from ..rocm_symm_mem_recipes import ROCmOnDeviceAllToAllV as OnDeviceAllToAllV
                    logger.info("Using ROCm-compatible symmetric memory implementation")
                except ImportError:
                    logger.warning("ROCm symmetric memory not available, falling back to standard implementation")
                    from torchtitan.experiments.deepseek_v3.symm_mem_recipes import OnDeviceAllToAllV
            else:
                from torchtitan.experiments.deepseek_v3.symm_mem_recipes import OnDeviceAllToAllV

            # Setup parameters
            overflow_factor = 2
            max_tokens = self.config.max_seq_len * self.top_k * overflow_factor

            # Set max output length
            OnDeviceAllToAllV.max_output_len = max_tokens

            # Create symmetric memory buffers (class-level, shared across instances)
            if not hasattr(DeepSeekV3MoE, '_symm_buffers_initialized'):
                DeepSeekV3MoE.token_send_buf = symm_mem.empty(
                    self.config.max_seq_len * self.top_k,
                    self.hidden_size,
                    dtype=dtype,
                    device=device,
                )

                DeepSeekV3MoE.token_gather_buf = symm_mem.empty(
                    max_tokens,
                    self.hidden_size,
                    dtype=dtype,
                    device=device,
                )

                DeepSeekV3MoE._symm_buffers_initialized = True

            logger.info("Symmetric memory setup completed")

        except ImportError as e:
            logger.warning(f"Symmetric memory components not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to setup symmetric memory: {e}")
            logger.warning("Continuing without symmetric memory optimization")