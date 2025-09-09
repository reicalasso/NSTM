# tests/test_token_router.py

import torch
import pytest
from src.nstm.core.token_router import TokenToStateRouter
from src.nstm.core.types import NSTMConfig

class TestTokenToStateRouter:
    """Unit tests for the TokenToStateRouter class."""
    
    def test_initialization(self, default_config):
        """Test that TokenToStateRouter initializes correctly."""
        router = TokenToStateRouter(default_config)
        
        assert hasattr(router, 'token_to_query')
        # Check linear layer input/output dimensions
        assert router.token_to_query.in_features == default_config.token_dim
        assert router.token_to_query.out_features == default_config.state_dim
        
    def test_compute_routing_weights(self, default_config):
        """Test routing weight computation."""
        router = TokenToStateRouter(default_config)
        batch_size = 3
        seq_len = 5
        num_states = 7
        
        tokens = torch.randn(batch_size, seq_len, default_config.token_dim)
        states = torch.randn(batch_size, num_states, default_config.state_dim)
        
        routing_weights = router.compute_routing_weights(tokens, states)
        
        # Check output shape
        assert routing_weights.shape == (batch_size, seq_len, num_states)
        
        # Check that weights are normalized (sum to 1 along the last dimension)
        weight_sums = routing_weights.sum(dim=-1)
        expected_sums = torch.ones(batch_size, seq_len)
        assert torch.allclose(weight_sums, expected_sums, atol=1e-6)
        
        # Check that all weights are positive (softmax output)
        assert torch.all(routing_weights >= 0)
        
    def test_route_tokens(self, default_config):
        """Test token routing functionality."""
        router = TokenToStateRouter(default_config)
        batch_size = 2
        seq_len = 4
        num_states = 6
        
        tokens = torch.randn(batch_size, seq_len, default_config.token_dim)
        states = torch.randn(batch_size, num_states, default_config.state_dim)
        
        routed_tokens, routing_weights = router.route_tokens(tokens, states)
        
        # Check that tokens are returned unchanged
        assert torch.allclose(routed_tokens, tokens)
        
        # Check routing weights shape and properties
        assert routing_weights.shape == (batch_size, seq_len, num_states)
        # Check normalization
        weight_sums = routing_weights.sum(dim=-1)
        expected_sums = torch.ones(batch_size, seq_len)
        assert torch.allclose(weight_sums, expected_sums, atol=1e-6)
        
    def test_forward(self, default_config):
        """Test forward method."""
        router = TokenToStateRouter(default_config)
        batch_size = 2
        seq_len = 4
        num_states = 6
        
        tokens = torch.randn(batch_size, seq_len, default_config.token_dim)
        states = torch.randn(batch_size, num_states, default_config.state_dim)
        
        routed_tokens, routing_weights = router(tokens, states)
        
        # Same checks as route_tokens
        assert torch.allclose(routed_tokens, tokens)
        assert routing_weights.shape == (batch_size, seq_len, num_states)
        weight_sums = routing_weights.sum(dim=-1)
        expected_sums = torch.ones(batch_size, seq_len)
        assert torch.allclose(weight_sums, expected_sums, atol=1e-6)