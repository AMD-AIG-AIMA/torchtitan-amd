#!/usr/bin/env python3
import torch
from torchtitan.models.moe import MoEArgs

def test_config_conversion():
    """Test MoEArgs to DeepSeek config conversion"""
    print("Testing config conversion...")
    
    from .hybrid_moe import create_deepseek_config
    
    moe_args = MoEArgs(
        num_experts=8,
        num_shared_experts=1,
        top_k=2,
        score_func="sigmoid",
        route_norm=True,
        route_scale=2.0,
    )
    
    config = create_deepseek_config(moe_args, dim=512, hidden_dim=2048, max_seq_len=1024)
    
    # Verify conversion
    assert config.n_routed_experts == 8
    assert config.n_shared_experts == 1
    assert config.num_experts_per_tok == 2
    assert config.hidden_size == 512
    assert config.moe_intermediate_size == 2048
    assert config.scoring_func == "sigmoid"
    assert config.norm_topk_prob == True
    assert config.routed_scaling_factor == 2.0
    
    print("✅ Config conversion works correctly")


def test_hybrid_moe_creation():
    """Test hybrid MoE creation and forward pass"""
    print("Testing hybrid MoE creation...")
    
    from .hybrid_moe import LLaMA4SymmMemMoE
    
    moe_args = MoEArgs(
        num_experts=4,
        num_shared_experts=1,
        top_k=2,
        score_func="sigmoid",
    )
    
    # Test standard MoE (no expert parallelism)
    hybrid_moe = LLaMA4SymmMemMoE(
        moe_args=moe_args,
        dim=256,
        hidden_dim=1024,
        ep_enabled=False,
    )
    
    print(f"Created: {hybrid_moe}")
    assert hybrid_moe.implementation_type == "standard"
    assert not hybrid_moe.is_symmetric_memory_enabled
    assert hybrid_moe.expert_parallel_size == 1
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    x = torch.randn(batch_size, seq_len, 256)
    
    output = hybrid_moe(x)
    assert output.shape == x.shape
    
    print("✅ Hybrid MoE creation and forward pass work correctly")


def test_deepseek_components():
    """Test individual DeepSeek V3 components"""
    print("Testing DeepSeek V3 components...")
    
    from .deepseek_v3_components import SimpleMLP, SimpleRouter, ROCmGroupGEMM
    
    # Test SimpleMLP
    mlp = SimpleMLP(hidden_size=256, intermediate_size=1024)
    x = torch.randn(10, 256)
    output = mlp(x)
    assert output.shape == (10, 256)
    print("✅ SimpleMLP works")
    
    # Test SimpleRouter
    router = SimpleRouter(hidden_size=256, num_experts=8, top_k=2, scoring_func="sigmoid")
    indices, weights = router(x)
    assert indices.shape == (10, 2)
    assert weights.shape == (10, 2)
    assert indices.dtype == torch.int32
    print("✅ SimpleRouter works")
    
    # Test ROCmGroupGEMM
    group_gemm = ROCmGroupGEMM()
    print(f"✅ ROCmGroupGEMM initialized (primus available: {group_gemm.primus_available})")
    
    print("✅ All DeepSeek V3 components work correctly")


def test_error_handling():
    """Test error handling and fallbacks"""
    print("Testing error handling...")
    
    from .hybrid_moe import LLaMA4SymmMemMoE
    
    moe_args = MoEArgs(num_experts=4, top_k=2)
    
    # Test with expert parallelism enabled but no distributed setup
    hybrid_moe = LLaMA4SymmMemMoE(
        moe_args=moe_args,
        dim=128,
        hidden_dim=512,
        ep_enabled=True,  # Should fall back to standard
    )
    
    # Should fall back to standard implementation
    assert hybrid_moe.implementation_type == "standard"
    print("✅ Graceful fallback to standard MoE works")
    
    # Test symmetric memory setup on standard MoE (should be no-op)
    hybrid_moe.setup_symmetric_memory(torch.float32, torch.device("cpu"))
    print("✅ Symmetric memory setup handles standard MoE gracefully")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("CLEANED HYBRID MOE TESTS")
    print("=" * 60)
    
    tests = [
        test_config_conversion,
        test_hybrid_moe_creation,
        test_deepseek_components,
        test_error_handling,
    ]
    
    passed = 0
    for test in tests:
        try:
            print(f"\n--- Running {test.__name__} ---")
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Cleaned implementations are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementations.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)