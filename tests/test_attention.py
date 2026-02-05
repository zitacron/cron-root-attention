"""
Cron Root Attention Tests
==========================

Test correctness and functionality of Cron Root Attention.
"""

import pytest
import torch
import torch.nn.functional as F
import math


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    pytest.skip("CUDA not available")


@pytest.fixture
def dtype():
    return torch.float16


class TestCronRootAttention:
    """Tests for Cron Root Attention functionality."""
    
    def test_import(self):
        """Test that imports work correctly."""
        from cron_root_attention import cron_root_attention
        from cron_root_attention import CronRootAttention
        from cron_root_attention import CronRootMultiheadAttention
        from cron_root_attention import get_gpu_info
        from cron_root_attention import GPU_SM_MAP
        assert callable(cron_root_attention)
        assert callable(get_gpu_info)
    
    def test_forward_shape(self, device, dtype):
        """Test output shape matches input shape."""
        from cron_root_attention import cron_root_attention
        
        B, H, S, D = 2, 4, 256, 64
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)
        
        output = cron_root_attention(q, k, v)
        
        assert output.shape == (B, H, S, D)
        assert output.dtype == dtype
    
    def test_no_nan_inf(self, device, dtype):
        """Test output contains no NaN or Inf values."""
        from cron_root_attention import cron_root_attention
        
        B, H, S, D = 1, 8, 512, 64
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)
        
        output = cron_root_attention(q, k, v)
        
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
    
    def test_backward_shape(self, device):
        """Test backward pass produces correct gradient shapes."""
        from cron_root_attention import cron_root_attention
        
        # Use FP32 for gradient testing
        dtype = torch.float32
        B, H, S, D = 1, 2, 128, 32
        
        q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        
        output = cron_root_attention(q, k, v)
        loss = output.sum()
        loss.backward()
        
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape
        assert v.grad.shape == v.shape
    
    def test_backward_no_nan(self, device):
        """Test backward pass produces no NaN gradients."""
        from cron_root_attention import cron_root_attention
        
        dtype = torch.float32
        B, H, S, D = 1, 2, 256, 32
        
        q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        
        output = cron_root_attention(q, k, v)
        grad_out = torch.randn_like(output)
        output.backward(grad_out)
        
        assert not torch.isnan(q.grad).any(), "dQ contains NaN"
        assert not torch.isnan(k.grad).any(), "dK contains NaN"
        assert not torch.isnan(v.grad).any(), "dV contains NaN"
    
    def test_module_api(self, device, dtype):
        """Test CronRootAttention module API."""
        from cron_root_attention import CronRootAttention
        
        B, S, D = 2, 256, 512
        n_heads = 8
        
        attn = CronRootAttention(d_model=D, n_heads=n_heads).to(device).to(dtype)
        x = torch.randn(B, S, D, device=device, dtype=dtype)
        
        output = attn(x)
        
        assert output.shape == (B, S, D)
    
    def test_multihead_attention_api(self, device, dtype):
        """Test CronRootMultiheadAttention module API."""
        from cron_root_attention import CronRootMultiheadAttention
        
        B, S, D = 2, 256, 512
        n_heads = 8
        
        mha = CronRootMultiheadAttention(
            embed_dim=D, num_heads=n_heads, batch_first=True
        ).to(device).to(dtype)
        x = torch.randn(B, S, D, device=device, dtype=dtype)
        
        output, attn_weights = mha(x, x, x)
        
        assert output.shape == (B, S, D)
        assert attn_weights is None  # Sparse attention doesn't return dense weights
    
    def test_gpu_info(self, device):
        """Test GPU info function."""
        from cron_root_attention import get_gpu_info
        
        info = get_gpu_info()
        
        assert 'gpu_name' in info
        assert 'sm_count' in info
        assert 'is_known_gpu' in info
        assert info['sm_count'] > 0
    
    def test_hybrid_backend(self, device, dtype):
        """Test hybrid attention backend selection."""
        from cron_root_attention import cron_root_attention_hybrid
        
        B, H, S, D = 1, 4, 2048, 64
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)
        
        # Test all backends
        out_auto = cron_root_attention_hybrid(q, k, v, backend="auto")
        out_cron = cron_root_attention_hybrid(q, k, v, backend="cron_root")
        out_sdpa = cron_root_attention_hybrid(q, k, v, backend="sdpa")
        
        assert out_auto.shape == (B, H, S, D)
        assert out_cron.shape == (B, H, S, D)
        assert out_sdpa.shape == (B, H, S, D)


class TestCronRootScaling:
    """Tests for √N scaling properties."""
    
    def test_complexity_scaling(self, device, dtype):
        """Test that timing scales sub-quadratically."""
        from cron_root_attention import cron_root_attention
        import time
        
        B, H, D = 1, 4, 64
        
        times = {}
        for S in [1024, 4096]:
            q = torch.randn(B, H, S, D, device=device, dtype=dtype)
            k = torch.randn(B, H, S, D, device=device, dtype=dtype)
            v = torch.randn(B, H, S, D, device=device, dtype=dtype)
            
            # Warmup
            for _ in range(3):
                _ = cron_root_attention(q, k, v)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(10):
                _ = cron_root_attention(q, k, v)
            torch.cuda.synchronize()
            times[S] = (time.perf_counter() - start) / 10
        
        # 4x sequence length should be < 16x time for O(N√N) vs O(N²)
        ratio = times[4096] / times[1024]
        assert ratio < 16, f"Scaling ratio {ratio:.1f} suggests quadratic complexity"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
