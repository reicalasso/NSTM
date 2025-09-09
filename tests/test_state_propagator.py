# tests/test_state_propagator.py

import torch
import pytest
from src.nstm.core.state_propagator import StatePropagator
from src.nstm.core.types import NSTMConfig

class TestStatePropagator:
    """Unit tests for the StatePropagator class."""
    
    def test_initialization_gru(self, default_config):
        """Test GRU type StatePropagator initialization."""
        config = default_config._replace(gate_type='gru')
        propagator = StatePropagator(config)
        
        assert hasattr(propagator, 'gate_layer')
        # For GRU: input_dim=state_dim*2, output_dim=state_dim*3
        expected_weight_shape = (config.state_dim * 3, config.state_dim * 2)
        assert propagator.gate_layer.weight.shape == expected_weight_shape
        
    def test_initialization_lstm(self, default_config):
        """Test LSTM type StatePropagator initialization."""
        config = default_config._replace(gate_type='lstm')
        propagator = StatePropagator(config)
        
        assert hasattr(propagator, 'gate_layer')
        # For LSTM: input_dim=state_dim*2, output_dim=state_dim*4
        expected_weight_shape = (config.state_dim * 4, config.state_dim * 2)
        assert propagator.gate_layer.weight.shape == expected_weight_shape
        
    def test_forward_single_state(self, default_config):
        """Test single state forward pass."""
        propagator = StatePropagator(default_config)
        batch_size = 3
        prev_state = torch.randn(batch_size, default_config.state_dim)
        input_token = torch.randn(batch_size, default_config.state_dim)
        
        new_state = propagator.forward_single_state(prev_state, input_token)
        
        assert new_state.shape == (batch_size, default_config.state_dim)
        # Check that output is not identical to input (transformation occurred)
        assert not torch.allclose(new_state, prev_state, atol=1e-5)
        
    def test_forward_multiple_states_without_routing(self, default_config):
        """Test multiple states forward pass without routing weights."""
        propagator = StatePropagator(default_config)
        batch_size = 2
        num_states = 5
        seq_len = 8
        
        prev_states = torch.randn(batch_size, num_states, default_config.state_dim)
        input_tokens = torch.randn(batch_size, seq_len, default_config.state_dim)
        
        updated_states = propagator.forward_multiple_states(prev_states, input_tokens)
        
        assert updated_states.shape == (batch_size, num_states, default_config.state_dim)
        # Check that output is not identical to input (transformation occurred)
        assert not torch.allclose(updated_states, prev_states, atol=1e-5)
        
    def test_forward_multiple_states_with_routing(self, default_config):
        """Test multiple states forward pass with routing weights."""
        propagator = StatePropagator(default_config)
        batch_size = 2
        num_states = 5
        seq_len = 8
        
        prev_states = torch.randn(batch_size, num_states, default_config.state_dim)
        input_tokens = torch.randn(batch_size, seq_len, default_config.state_dim)
        routing_weights = torch.rand(batch_size, seq_len, num_states)
        # Normalize to make it a probability distribution over states for each token
        routing_weights = torch.softmax(routing_weights, dim=-1)
        
        updated_states = propagator.forward_multiple_states(
            prev_states, input_tokens, routing_weights
        )
        
        assert updated_states.shape == (batch_size, num_states, default_config.state_dim)
        # Check that output is not identical to input (transformation occurred)
        assert not torch.allclose(updated_states, prev_states, atol=1e-5)
        
    def test_forward_auto_single_multiple(self, default_config):
        """Test automatic single/multiple state handling in main forward."""
        propagator = StatePropagator(default_config)
        batch_size = 2
        seq_len = 6
        
        # Test with single state
        prev_states_single = torch.randn(batch_size, 1, default_config.state_dim)
        input_tokens_single = torch.randn(batch_size, seq_len, default_config.state_dim)
        
        updated_states_single = propagator(prev_states_single, input_tokens_single)
        assert updated_states_single.shape == (batch_size, 1, default_config.state_dim)
        
        # Test with multiple states
        num_states = 4
        prev_states_multi = torch.randn(batch_size, num_states, default_config.state_dim)
        input_tokens_multi = torch.randn(batch_size, seq_len, default_config.state_dim)
        
        updated_states_multi = propagator(prev_states_multi, input_tokens_multi)
        assert updated_states_multi.shape == (batch_size, num_states, default_config.state_dim)