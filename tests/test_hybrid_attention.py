# tests/test_hybrid_attention.py

import torch
import pytest
from src.nstm.core.attention import HybridAttention
from src.nstm.core.types import NSTMConfig

class TestHybridAttention:
    """Unit tests for the HybridAttention class."""
    
    def test_initialization(self, default_config):
        """Test that HybridAttention initializes correctly."""
        attention = HybridAttention(default_config)
        
        # Check that all linear layers are created
        assert hasattr(attention, 'token_to_state_q')
        assert hasattr(attention, 'token_to_state_k')
        assert hasattr(attention, 'token_to_state_v')
        assert hasattr(attention, 'state_to_state_q')
        assert hasattr(attention, 'state_to_state_k')
        assert hasattr(attention, 'state_to_state_v')
        assert hasattr(attention, 'output_projection')
        
        # Check dimensions
        state_dim = default_config.state_dim
        token_dim = default_config.token_dim
        assert attention.token_to_state_q.in_features == token_dim
        assert attention.token_to_state_q.out_features == state_dim
        assert attention.state_to_state_q.in_features == state_dim
        assert attention.state_to_state_q.out_features == state_dim
        assert attention.output_projection.in_features == state_dim
        assert attention.output_projection.out_features == state_dim
        
    def test_split_combine_heads(self, default_config):
        """Test multi-head split and combine operations."""
        attention = HybridAttention(default_config)
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, default_config.state_dim)
        
        # Split heads
        x_split = attention._split_heads(x)
        expected_shape = (
            batch_size, 
            default_config.num_attention_heads, 
            seq_len, 
            default_config.state_dim // default_config.num_attention_heads
        )
        assert x_split.shape == expected_shape
        
        # Combine heads
        x_combined = attention._combine_heads(x_split)
        assert x_combined.shape == (batch_size, seq_len, default_config.state_dim)
        # Check that combining split tensors gives the original back
        assert torch.allclose(x, x_combined, atol=1e-6)
        
    def test_token_to_state_attention(self, default_config):
        """Test token-to-state attention mechanism."""
        attention = HybridAttention(default_config)
        batch_size = 2
        seq_len = 4
        num_states = 6
        
        tokens = torch.randn(batch_size, seq_len, default_config.token_dim)
        states = torch.randn(batch_size, num_states, default_config.state_dim)
        
        attn_output, attn_weights = attention.token_to_state_attention(tokens, states)
        
        # Check output shape
        assert attn_output.shape == (batch_size, num_states, default_config.state_dim)
        
        # Check attention weights shape
        expected_weight_shape = (
            batch_size, 
            default_config.num_attention_heads, 
            num_states, 
            seq_len
        )
        assert attn_weights.shape == expected_weight_shape
        
        # Check that weights are normalized
        weight_sums = attn_weights.sum(dim=-1)
        expected_sums = torch.ones(
            batch_size, 
            default_config.num_attention_heads, 
            num_states
        )
        assert torch.allclose(weight_sums, expected_sums, atol=1e-6)
        
    def test_state_to_state_attention(self, default_config):
        """Test state-to-state attention mechanism."""
        attention = HybridAttention(default_config)
        batch_size = 3
        num_states = 5
        
        states = torch.randn(batch_size, num_states, default_config.state_dim)
        
        attn_output, attn_weights = attention.state_to_state_attention(states)
        
        # Check output shape
        assert attn_output.shape == (batch_size, num_states, default_config.state_dim)
        
        # Check attention weights shape
        expected_weight_shape = (
            batch_size, 
            default_config.num_attention_heads, 
            num_states, 
            num_states
        )
        assert attn_weights.shape == expected_weight_shape
        
        # Check that weights are normalized
        weight_sums = attn_weights.sum(dim=-1)
        expected_sums = torch.ones(
            batch_size, 
            default_config.num_attention_heads, 
            num_states
        )
        assert torch.allclose(weight_sums, expected_sums, atol=1e-6)
        
    def test_forward(self, default_config):
        """Test full hybrid attention forward pass."""
        attention = HybridAttention(default_config)
        batch_size = 2
        seq_len = 4
        num_states = 6
        
        tokens = torch.randn(batch_size, seq_len, default_config.token_dim)
        states = torch.randn(batch_size, num_states, default_config.state_dim)
        
        output, ts_weights, ss_weights = attention(tokens, states)
        
        # Check output shape
        assert output.shape == (batch_size, num_states, default_config.state_dim)
        
        # Check token-to-state weights shape
        expected_ts_shape = (
            batch_size, 
            default_config.num_attention_heads, 
            num_states, 
            seq_len
        )
        assert ts_weights.shape == expected_ts_shape
        
        # Check state-to-state weights shape
        expected_ss_shape = (
            batch_size, 
            default_config.num_attention_heads, 
            num_states, 
            num_states
        )
        assert ss_weights.shape == expected_ss_shape