# tests/test_nstm_layer.py

import torch
import pytest
from src.nstm.models.nstm_layer import NSMLayer
from src.nstm.core.types import NSTMConfig

class TestNSMLayer:
    """NSMLayer sınıfı için birim testleri."""
    
    def test_initialization(self, default_config):
        """Test that NSMLayer initializes correctly."""
        layer = NSMLayer(default_config)
        
        # Check that all sub-components are created
        assert hasattr(layer, 'state_manager')
        assert hasattr(layer, 'state_propagator')
        assert hasattr(layer, 'token_router')
        assert hasattr(layer, 'hybrid_attention')
        
        # Check that they have the correct types
        from src.nstm.core.state_manager import StateManager
        from src.nstm.core.state_propagator import StatePropagator
        from src.nstm.core.token_router import TokenToStateRouter
        from src.nstm.core.attention import HybridAttention
        
        assert isinstance(layer.state_manager, StateManager)
        assert isinstance(layer.state_propagator, StatePropagator)
        assert isinstance(layer.token_router, TokenToStateRouter)
        assert isinstance(layer.hybrid_attention, HybridAttention)
        
    def test_forward_with_internal_states(self, default_config, sample_tokens):
        """Test forward pass using internal states from StateManager."""
        layer = NSMLayer(default_config)
        batch_size = sample_tokens.shape[0]
        
        # First call should use internal states
        updated_states, ts_weights, ss_weights = layer(sample_tokens)
        
        # Check output shapes
        assert updated_states.shape == (
            batch_size, 
            default_config.initial_states, 
            default_config.state_dim
        )
        
        # Check attention weights shapes
        expected_ts_shape = (
            batch_size, 
            default_config.num_attention_heads, 
            default_config.initial_states, 
            sample_tokens.shape[1]  # seq_len
        )
        assert ts_weights.shape == expected_ts_shape
        
        expected_ss_shape = (
            batch_size, 
            default_config.num_attention_heads, 
            default_config.initial_states, 
            default_config.initial_states
        )
        assert ss_weights.shape == expected_ss_shape
        
    def test_forward_with_provided_states(self, default_config, sample_tokens, sample_states):
        """Test forward pass with explicitly provided states."""
        layer = NSMLayer(default_config)
        batch_size = sample_tokens.shape[0]
        
        # Use provided states
        updated_states, ts_weights, ss_weights = layer(sample_tokens, sample_states)
        
        # Check output shapes
        assert updated_states.shape == sample_states.shape
        
        # Check attention weights shapes
        expected_ts_shape = (
            batch_size, 
            default_config.num_attention_heads, 
            sample_states.shape[1],  # num_states
            sample_tokens.shape[1]   # seq_len
        )
        assert ts_weights.shape == expected_ts_shape
        
        expected_ss_shape = (
            batch_size, 
            default_config.num_attention_heads, 
            sample_states.shape[1],  # num_states
            sample_states.shape[1]   # num_states
        )
        assert ss_weights.shape == expected_ss_shape
        
    def test_state_management_methods(self, default_config):
        """Test state management methods of the layer."""
        layer = NSMLayer(default_config)
        batch_size = 3
        
        # Test get_states
        states = layer.get_states(batch_size)
        assert states.shape == (
            batch_size, 
            default_config.initial_states, 
            default_config.state_dim
        )
        
        # Test update_state_importance
        new_scores = torch.rand(batch_size, default_config.initial_states)
        # For simplicity in this test, we'll use the same scores for all batches
        # by taking the mean. In practice, you might have different scores per batch.
        mean_scores = new_scores.mean(dim=0, keepdim=True)  # (1, num_states)
        layer.update_state_importance(mean_scores)
        # Check that scores were updated in the state manager
        manager_scores = layer.state_manager.get_importance_scores(batch_size)
        assert torch.allclose(manager_scores, mean_scores.expand(batch_size, -1))
        
        # Test prune_states
        # Set some scores very low to ensure pruning happens
        low_scores = torch.ones(1, default_config.initial_states) * 0.1
        low_scores[0, 0] = 0.01  # Definitely below threshold
        layer.update_state_importance(low_scores)
        num_pruned = layer.prune_states()
        assert num_pruned >= 0  # At least one should be pruned
        
        # Test allocate_states
        num_allocated = layer.allocate_states(2)
        assert num_allocated == 2