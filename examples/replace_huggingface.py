"""
Replace HuggingFace Model Attention with Cron Root
====================================================

Example of patching a HuggingFace model to use Cron Root Attention.
"""

import torch
import torch.nn as nn
from cron_root_attention import cron_root_attention


def patch_llama_attention(model):
    """
    Patch Llama model attention layers to use Cron Root Attention.
    
    This example shows how to replace the attention computation
    in a HuggingFace Llama model with Cron Root's O(N√N) attention.
    
    Args:
        model: A HuggingFace LlamaForCausalLM model
        
    Returns:
        The patched model
    """
    from transformers.models.llama.modeling_llama import LlamaAttention
    
    original_forward = LlamaAttention.forward
    
    def patched_forward(self, hidden_states, attention_mask=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        
        # Get Q, K, V projections (standard Llama code)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Repeat K/V for GQA
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Use Cron Root Attention instead of standard attention!
        # This gives O(N√N) complexity instead of O(N²)
        attn_output = cron_root_attention(query_states, key_states, value_states)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, None
    
    # Apply patch
    LlamaAttention.forward = patched_forward
    print("✓ Patched Llama attention with Cron Root Attention")
    
    return model


if __name__ == "__main__":
    print("Cron Root HuggingFace Integration Example")
    print("=" * 50)
    print()
    print("Usage:")
    print("  from transformers import AutoModelForCausalLM")
    print("  model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')")
    print("  patch_llama_attention(model)")
    print()
    print("This will replace all attention layers with O(N√N) Cron Root Attention.")
