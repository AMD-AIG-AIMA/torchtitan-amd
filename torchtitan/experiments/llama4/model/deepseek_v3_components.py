from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import is_hip

import torch.distributed._symmetric_memory as symm_mem
from ..rocm_symm_mem_recipes import OnDeviceAllToAllV
from torchtitan.experiments.kernels.moe.indices import generate_permute_indices


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


class RandomSTE(torch.autograd.Function):
    """
    Straight-Through Estimator(STE) function that returns random values
    with different seed for each rank.

    This is used to generate random logits of router for load-balanced benchmark.
    """

    generator = None

    @staticmethod
    def forward(ctx, logits):
        """
        Forward pass returns random logits with rank-specific seed.
        """
        if RandomSTE.generator is None:
            global_rank = torch.distributed.get_rank()
            base_seed = 42
            seed = base_seed + global_rank
            RandomSTE.generator = torch.Generator(device=logits.device)
            RandomSTE.generator.manual_seed(seed)

        random_logits = logits.clone().normal_(generator=RandomSTE.generator)
        return random_logits

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass propagates the gradient for logits.
        """
        return grad_output


def apply_random_logits(logits):
    """
    Apply the RandomSTE function to the logits.
    """
    return RandomSTE.apply(logits)


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


class TokenChoiceTopKRouter(nn.Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        score_func (Literal["softmax", "sigmoid"]): Whether to use sigmoid or softmax for router scores.
        route_norm (bool): Whether to normalize the routing scores when using sigmoid.
        route_scale (float): Scaling factor applied to the routing scores.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"],
        route_norm: bool,
        route_scale: float,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        scores = self.gate(x)
        scores = apply_random_logits(scores)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_function}")

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        if expert_bias is not None:
            _, selected_experts_indices = torch.topk(
                scores + expert_bias, k=self.top_k, dim=1
            )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.top_k, dim=1
            )

        if self.score_func == "sigmoid" and self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_scores, selected_experts_indices, num_tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


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

    # Symmetric memory buffers shared by all MoE instances across layers
    token_send_buf: Optional[torch.Tensor] = None
    token_gather_buf: Optional[torch.Tensor] = None
    _symm_mem_buffer_allocated: bool = False

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        logger.info(f"Initializing DeepSeekV3MoE with {config}")
        # Setup expert parallelism
        self._setup_expert_parallelism()

        # Create local experts
        self._create_experts()

        # Create router
        self.router = TokenChoiceTopKRouter(
            dim=config.hidden_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            score_func=config.score_func,
            route_norm=config.route_norm,
            route_scale=config.route_scale,
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
        else:
            self.ep_size = 1
            self.ep_rank = 0
            self.ep_group = None

        self.experts_per_rank = self.num_experts // self.ep_size

    def _create_experts(self):
        """Create expert networks for this rank"""
        self.experts = nn.ModuleList()
        
        for i in range(self.experts_per_rank):
            expert = SimpleMLP(self.hidden_size, self.intermediate_size)
            self.experts.append(expert)

    def sort_tokens(self, x, topk_ids, topk_weights):
        # This part sorts the token indices so that tokens routed to the same expert reside consecutively.
        # An implication is that tokens to the same "expert group" (i.e., device) are also consecutive.
        # Since this is an "aritificial" index creation (final outcome being
        # `idxs`), we don't need gradients here.

        with torch.no_grad():
            # [seq_len, num_experts]
            expert_counts = topk_ids.new_zeros(
                (topk_ids.shape[0], self.num_experts)
            )
            # Fill 1 to the selected experts
            expert_counts.scatter_(1, topk_ids, 1)
            tokens_per_expert = expert_counts.sum(dim=0)
            # Token indices for each expert
            token_indices = topk_ids.view(-1).argsort()

        sorted_tokens = x[token_indices // topk_ids.shape[1]]
        # assert sorted_tokens.shape == sorted_tokens_shape

        return (sorted_tokens, token_indices, tokens_per_expert)

    def get_send_buf(self, batch_size, seq_len):
        # [Why detach?] During a first forward-backward step, the buffer would
        # be included in a computational graph. In a second step, autograd will
        # return an error saying "Trying to backward through the graph a second
        # time (or directly access saved tensors more than once)". This is
        # because the buffer is still in the graph, and autograd is trying to
        # backward through the graph a second time. To avoid this, we detach the
        # buffer from the graph. `detach()` returns a new tensor, which shares
        # the same storage with the original one.
        if not DeepSeekV3MoE._symm_mem_buffer_allocated:
            self.allocate_symm_mem_buffers(batch_size, seq_len)
        self.token_send_buf.grad = None
        return self.token_send_buf.detach()

    def get_gather_buf(self, batch_size, seq_len):
        # See [Why detach?] in `get_send_buf`
        if not DeepSeekV3MoE._symm_mem_buffer_allocated:
            self.allocate_symm_mem_buffers(batch_size, seq_len)
        self.token_gather_buf.grad = None
        return self.token_gather_buf.detach()

    def _run_group_gemm(self, contig_tokens, m_sizes, m_offsets):
        """Run the appropriate group GEMM implementation based on configuration"""
        return self.group_gemm_strategy.execute(
                contig_tokens, m_sizes, m_offsets, self
        )

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
        topk_weight, topk_ids, _ = self.router(x)

        (
            sorted_tokens,
            token_indices,
            tokens_per_expert,
        ) = self.sort_tokens(x, topk_ids, topk_weight)

        # keep the seqlen dimension for later use without holding onto the sorted tokens
        seqlen_sorted_tokens = sorted_tokens.shape[0]

        # This part exchange the information about the number of tokens send and
        # received by each expert. We can understand this information as "side
        # band", which is not part of the actual data. Thus no gradient is
        # needed.

        # Sum the tokens over local experts, then we get tokens per EP rank,
        # which is the input splits
        with torch.no_grad():
            tokens_per_expert_group = tokens_per_expert.new_empty(
                tokens_per_expert.shape[0]
            )
            dist.all_to_all_single(
                tokens_per_expert_group, tokens_per_expert, group=self.ep_group
            )
            input_splits = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)

        # Move input to the `token_send_buf` symm mem
        token_send_buf = self.get_send_buf(batch_size, seq_len)
        token_send_buf[: token_indices.shape[0]].copy_(sorted_tokens)
        # Note: `out=` avoids copy, but it is not differentiable
        # torch.index_select(x, 0, idxs // topk_ids.shape[1], out=token_send_buf[: idxs.shape[0]])
        token_gather_buf, output_splits = OnDeviceAllToAllV.apply(
            token_send_buf,
            input_splits,
            self.ep_group,
        )

        # We need to permute the received tokens so that tokens for the same expert are contiguous.
        # This part prepares a 1D tensor `permuted_indices` for such permutation.
        # This part doesn't need gradient.
        with torch.no_grad():
            permuted_indices, m_sizes, m_offsets = generate_permute_indices(
                tokens_per_expert_group,
                self.experts_per_rank,
                self.ep_size,
                token_gather_buf.shape[0],
                128,
            )

        # Permute the received tokens so that tokens for the same expert are contiguous.
        contig_tokens = token_gather_buf[permuted_indices]

        # group gemm - handle all three group gemms (up, gate, down for all experts)
        hidden_outputs = self._run_group_gemm(
            contig_tokens,
            m_sizes,
            m_offsets,
        )

        # Prepare buffer for tokens processed by experts
        processed_tokens = self.get_gather_buf(batch_size, seq_len)

        # Move into Symmetric Memory for the return shuffle
        processed_tokens[permuted_indices] = hidden_outputs

        # Now shuffle the tokens back to their original owner, i.e. EP to DP shuffle.
        # The input/output splits are just a reverse of the previous shuffle.
        token_return_buf, _ = OnDeviceAllToAllV.apply(
            processed_tokens,
            output_splits,
            self.ep_group,
        )

        returned_tokens = token_return_buf[:seqlen_sorted_tokens]
        output_tokens = torch.empty_like(returned_tokens)
        output_tokens[token_indices] = returned_tokens

        final_out = (
            output_tokens.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(returned_tokens.dtype)
        )

        final_out = final_out.view(*identity.shape)
        # Add shared expert contribution
        if self.shared_experts is not None:
            final_out = final_out + self.shared_experts(identity)

        return final_out

    def _combine_experts(self):
        """Create a mock module with arranged expert weights for grouped GEMM"""
        # Collect weights from all experts
        gate_weights = []
        up_weights = []
        down_weights = []
        
        for expert in self.experts:
            gate_weights.append(expert.gate_proj.weight)
            up_weights.append(expert.up_proj.weight)  
            down_weights.append(expert.down_proj.weight)
        
        self.gate_proj_weight = self.group_gemm_strategy.arrange_expert_weights(gate_weights, "gate_proj", self)
        self.up_proj_weight = self.group_gemm_strategy.arrange_expert_weights(up_weights, "up_proj", self)
        self.down_proj_weight = self.group_gemm_strategy.arrange_expert_weights(down_weights, "down_proj", self)
        

    def allocate_symm_mem_buffers(self, batch_size: int, seq_len: int):
        if DeepSeekV3MoE._symm_mem_buffer_allocated:
            return
        
        if self.ep_size == 1:
            logger.info("Symmetric memory not needed for ep_size=1")
            return

        # Setup parameters
        overflow_factor = 2
        total_tokens = batch_size * seq_len * self.top_k
        max_tokens = total_tokens * overflow_factor

        OnDeviceAllToAllV.max_output_len = max_tokens

        # Allocate send buffer
        if DeepSeekV3MoE.token_send_buf is None:
            DeepSeekV3MoE.token_send_buf = symm_mem.empty(
                total_tokens,
                self.hidden_size,
                dtype=self.symm_mem_dtype,
                device=self.symm_mem_device,
            )
            logger.info(f"Allocated token_send_buf with shape {DeepSeekV3MoE.token_send_buf.shape}")

        # Allocate gather buffer
        if DeepSeekV3MoE.token_gather_buf is None:
            DeepSeekV3MoE.token_gather_buf = symm_mem.empty(
                max_tokens,
                self.hidden_size,
                dtype=self.symm_mem_dtype,
                device=self.symm_mem_device,
            )
            logger.info(f"Allocated token_gather_buf with shape {DeepSeekV3MoE.token_gather_buf.shape}")

        DeepSeekV3MoE._symm_mem_buffer_allocated = True


    def setup_symm_mem(self, dtype: torch.dtype, device: torch.device):
        """Setup symmetric memory for distributed processing"""
        if self.ep_size == 1:
            logger.info("Symmetric memory not needed for ep_size=1")
            return
        
        self.symm_mem_dtype = dtype
        self.symm_mem_device = device
        self._combine_experts()
