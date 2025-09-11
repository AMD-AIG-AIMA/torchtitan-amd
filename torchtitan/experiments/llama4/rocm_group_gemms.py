import torch
import torch.nn.functional as F
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import is_hip

# Import the existing ROCm function from standard MoE
try:
    from torchtitan.models.moe import _run_experts_grouped_mm_rocm
    STANDARD_ROCM_AVAILABLE = True
except ImportError:
    logger.warning("Standard ROCm grouped GEMM function not available")
    STANDARD_ROCM_AVAILABLE = False

# Import DeepSeek V3 GroupGEMMStrategy base class
# try:
#     from torchtitan.experiments.deepseek_v3.group_gemms import GroupGEMMStrategy
#     DEEPSEEK_GROUP_GEMM_AVAILABLE = True
# except ImportError:
#     # Create a minimal base class if DeepSeek not available
#     class GroupGEMMStrategy:
#         def __init__(self, custom_activation):
#             self.activation_function = custom_activation
        
#         def arrange_expert_weights(self, all_weights, submod_name, module):
#             raise NotImplementedError("Requires arrange_expert_weights method")
        
#         def execute(self, contig_tokens, m_sizes, m_offsets, module):
#             raise NotImplementedError("GroupGEMM strategy must implement execute method")
        
#         @staticmethod
#         def is_available() -> bool:
#             return False
    
#     DEEPSEEK_GROUP_GEMM_AVAILABLE = False

class GroupGEMMStrategy:
    def __init__(self, custom_activation):
        self.activation_function = custom_activation
    
    def arrange_expert_weights(self, all_weights, submod_name, module):
        raise NotImplementedError("Requires arrange_expert_weights method")
    
    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        raise NotImplementedError("GroupGEMM strategy must implement execute method")
    
    @staticmethod
    def is_available() -> bool:
        return False


class ROCmGroupGEMMStrategy(GroupGEMMStrategy):
    """
    ROCm-optimized group GEMM strategy using primus_turbo.
    
    This strategy adapts the existing _run_experts_grouped_mm_rocm function
    to work with DeepSeek V3's Group GEMM interface, providing optimal
    performance on AMD GPUs while maintaining compatibility.
    """

    def __init__(self, custom_activation):
        super().__init__(custom_activation)
        self.primus_available = self._check_primus_availability()

    def _check_primus_availability(self):
        """Check if primus_turbo is available for ROCm"""
        if not is_hip():
            return False
        
        try:
            import primus_turbo.pytorch as turbo
            return True
        except ImportError:
            logger.warning("primus_turbo not available for ROCm grouped GEMM")
            return False

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """
        Arrange expert weights in the format expected by _run_experts_grouped_mm_rocm.
        
        The standard ROCm function expects weights in shape [num_experts, out_features, in_features],
        which is exactly how DeepSeek V3 stores individual expert weights.
        
        Args:
            all_weights: List of weight tensors from each expert
            submod_name: Name of the submodule ('gate_proj', 'up_proj', 'down_proj')
            module: The parent module that will store the arranged weights
            
        Returns:
            Combined weight tensor [num_experts, out_features, in_features]
        """
        if not all_weights:
            raise ValueError(f"No expert weights provided for {submod_name}")
        
        # Stack weights into the format expected by _run_experts_grouped_mm_rocm
        # Shape: [num_experts, out_features, in_features]
        combined_weights = torch.stack(all_weights, dim=0)
        
        logger.info(f"Arranged {submod_name} weights: {combined_weights.shape} for ROCm grouped GEMM")
        return combined_weights

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """
        Execute ROCm-optimized grouped GEMM operations.
        
        This method adapts the DeepSeek V3 grouped GEMM interface to use
        the existing _run_experts_grouped_mm_rocm function.
        
        Args:
            contig_tokens: Input tokens arranged contiguously by expert [total_tokens, hidden_size]
            m_sizes: Number of tokens for each expert [num_experts]
            m_offsets: Offset positions for each expert [num_experts] (unused in ROCm impl)
            module: MoE module containing the combined expert weights
            
        Returns:
            Processed tokens from all experts [total_tokens, hidden_size]
        """
        try:
            return self._run_rocm_grouped_gemm(contig_tokens, m_sizes, module)
        except Exception as e:
            logger.warning(f"ROCm grouped GEMM failed: {e}, falling back to manual loop")
            return self._run_manual_fallback(contig_tokens, m_sizes, module)

    def _run_rocm_grouped_gemm(self, contig_tokens, m_sizes, module):
        """
        Execute using the existing _run_experts_grouped_mm_rocm function.
        
        This is the main ROCm optimization path that uses primus_turbo
        for maximum performance on AMD GPUs.
        """
        if not STANDARD_ROCM_AVAILABLE:
            raise RuntimeError("Standard ROCm grouped GEMM function not available")
        
        if not self.primus_available:
            raise RuntimeError("primus_turbo not available")
        
        # Get combined expert weights from the module
        # These should have been arranged by arrange_expert_weights()
        if not (hasattr(module, 'gate_proj_weight') and 
                hasattr(module, 'up_proj_weight') and 
                hasattr(module, 'down_proj_weight')):
            raise RuntimeError("Module missing combined expert weights (gate_proj_weight, up_proj_weight, down_proj_weight)")
        
        w1 = module.gate_proj_weight  # [num_experts, intermediate_size, hidden_size]
        w2 = module.down_proj_weight  # [num_experts, hidden_size, intermediate_size]  
        w3 = module.up_proj_weight    # [num_experts, intermediate_size, hidden_size]
        
        # Ensure token count compatibility
        num_tokens_per_expert = m_sizes.to(torch.int64)
        
        # Validate shapes
        expected_experts = w1.shape[0]
        if len(num_tokens_per_expert) != expected_experts:
            raise RuntimeError(f"Mismatch: {len(num_tokens_per_expert)} expert token counts vs {expected_experts} expert weights")
        
        # Call the existing optimized ROCm function
        logger.debug(f"Running ROCm grouped GEMM: {contig_tokens.shape} tokens across {expected_experts} experts")
        
        output = _run_experts_grouped_mm_rocm(
            w1=w1,
            w2=w2, 
            w3=w3,
            x=contig_tokens,
            num_tokens_per_expert=num_tokens_per_expert,
        )
        
        logger.debug(f"ROCm grouped GEMM completed: {output.shape}")
        return output

    def _run_manual_fallback(self, contig_tokens, m_sizes, module):
        """
        Manual loop fallback when ROCm grouped GEMM is not available.
        
        This provides a functional fallback that processes each expert
        individually using standard PyTorch operations.
        """
        logger.info("Using manual loop fallback for expert processing")
        
        outputs = []
        offset = 0
        
        # Process each expert individually
        for expert_idx, num_tokens in enumerate(m_sizes.tolist()):
            if num_tokens == 0:
                continue
            
            # Get tokens for this expert
            expert_tokens = contig_tokens[offset:offset + num_tokens]
            
            # Get expert weights (assuming individual expert access)
            if hasattr(module, 'experts') and len(module.experts) > expert_idx:
                expert = module.experts[expert_idx]
                expert_output = expert(expert_tokens)
            else:
                # Use combined weights if individual experts not available
                w1 = module.gate_proj_weight[expert_idx]  # [intermediate_size, hidden_size]
                w2 = module.down_proj_weight[expert_idx] # [hidden_size, intermediate_size]
                w3 = module.up_proj_weight[expert_idx]   # [intermediate_size, hidden_size]
                
                # Manual SwiGLU computation
                h1 = F.linear(expert_tokens, w1)  # gate projection
                h3 = F.linear(expert_tokens, w3)  # up projection
                h = self.activation_function(h1) * h3  # SwiGLU
                expert_output = F.linear(h, w2)  # down projection
            
            outputs.append(expert_output)
            offset += num_tokens
        
        # Concatenate all expert outputs
        if outputs:
            result = torch.cat(outputs, dim=0)
            logger.debug(f"Manual fallback completed: {result.shape}")
            return result
        else:
            # No tokens processed, return empty tensor
            return torch.zeros(0, contig_tokens.shape[-1], 
                             device=contig_tokens.device, dtype=contig_tokens.dtype)

    @staticmethod
    def is_available() -> bool:
        """Check if ROCm grouped GEMM strategy is available"""
        return (
            is_hip() and 
            STANDARD_ROCM_AVAILABLE and 
            ROCmGroupGEMMStrategy._check_static_primus_availability()
        )
    
    @staticmethod
    def _check_static_primus_availability():
        """Static check for primus_turbo availability"""
        try:
            import primus_turbo.pytorch as turbo
            return True
        except ImportError:
            return False


class TorchGroupGEMMFallback(GroupGEMMStrategy):
    """
    Fallback Group GEMM strategy using PyTorch's grouped matrix multiplication.
    
    This provides a fallback when ROCm-specific optimizations are not available
    but torch._grouped_mm is still functional.
    """

    def __init__(self, custom_activation):
        super().__init__(custom_activation)

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Arrange weights for torch._grouped_mm"""
        if not all_weights:
            raise ValueError(f"No expert weights provided for {submod_name}")
        
        # Stack weights in the same format as ROCm
        combined_weights = torch.stack(all_weights, dim=0)
        return combined_weights

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """Execute using PyTorch grouped matrix multiplication"""
        try:
            return self._run_torch_grouped_gemm(contig_tokens, m_sizes, module)
        except Exception as e:
            logger.warning(f"Torch grouped GEMM failed: {e}, falling back to manual loop")
            return self._run_manual_loop(contig_tokens, m_sizes, module)

    def _run_torch_grouped_gemm(self, contig_tokens, m_sizes, module):
        """Use torch._grouped_mm for grouped operations"""
        
        # Get combined weights
        w1 = module.gate_proj_weight   # [num_experts, intermediate_size, hidden_size]
        w2 = module.down_proj_weight   # [num_experts, hidden_size, intermediate_size]
        w3 = module.up_proj_weight     # [num_experts, intermediate_size, hidden_size]
        
        # Compute cumulative offsets for grouped_mm
        offsets = torch.cumsum(m_sizes, dim=0, dtype=torch.int32)
        
        # Gate projection
        h1 = torch._grouped_mm(
            contig_tokens.bfloat16(),
            w1.bfloat16().transpose(-2, -1),  # Transpose for grouped_mm
            offs=offsets
        )
        
        # Up projection
        h3 = torch._grouped_mm(
            contig_tokens.bfloat16(),
            w3.bfloat16().transpose(-2, -1),
            offs=offsets
        )
        
        # Apply SwiGLU activation
        h = self.activation_function(h1) * h3
        
        # Down projection
        output = torch._grouped_mm(
            h,
            w2.bfloat16().transpose(-2, -1),
            offs=offsets
        ).type_as(contig_tokens)
        
        return output

    def _run_manual_loop(self, contig_tokens, m_sizes, module):
        """Manual loop fallback identical to ROCm version"""
        outputs = []
        offset = 0
        
        for expert_idx, num_tokens in enumerate(m_sizes.tolist()):
            if num_tokens == 0:
                continue
            
            expert_tokens = contig_tokens[offset:offset + num_tokens]
            
            w1 = module.gate_proj_weight[expert_idx]
            w2 = module.down_proj_weight[expert_idx]
            w3 = module.up_proj_weight[expert_idx]
            
            h1 = F.linear(expert_tokens, w1)
            h3 = F.linear(expert_tokens, w3)
            h = self.activation_function(h1) * h3
            expert_output = F.linear(h, w2)
            
            outputs.append(expert_output)
            offset += num_tokens
        
        return torch.cat(outputs, dim=0) if outputs else torch.zeros(
            0, contig_tokens.shape[-1], device=contig_tokens.device, dtype=contig_tokens.dtype
        )

    @staticmethod
    def is_available() -> bool:
        """Torch grouped MM is generally available"""
        return True