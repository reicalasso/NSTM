# tests/test_state_manager.py

import torch
import pytest
from src.nstm.core.state_manager import StateManager
from src.nstm.core.types import NSTMConfig

class TestStateManager:
    """StateManager sınıfı için birim testleri."""
    
    def test_initialization(self, default_config):
        """Test that StateManager initializes correctly."""
        manager = StateManager(default_config)
        
        # Check that states and importance scores are initialized
        assert hasattr(manager, 'states')
        assert hasattr(manager, 'importance_scores')
        
        # Check shapes
        assert manager.states.shape == (1, default_config.initial_states, default_config.state_dim)
        assert manager.importance_scores.shape == (1, default_config.initial_states)
        
        # Check that importance scores are initialized to ones
        assert torch.all(manager.importance_scores == 1.0)
    
    def test_create_states(self, default_config):
        """Test state creation functionality."""
        manager = StateManager(default_config)
        num_states = 5
        batch_size = 3
        
        new_states = manager.create_states(num_states, batch_size)
        
        assert new_states.shape == (batch_size, num_states, default_config.state_dim)
        # Check that states are initialized with random values (not all zeros)
        assert not torch.all(new_states == 0)
        
    def test_get_states(self, default_config):
        """Test getting states with different batch sizes."""
        manager = StateManager(default_config)
        batch_size = 1
        states = manager.get_states(batch_size)
        
        assert states.shape == (batch_size, default_config.initial_states, default_config.state_dim)
        assert torch.allclose(states, manager.states) # Should be the same for batch_size=1
        
        # Test with larger batch size (broadcasting)
        batch_size = 4
        states = manager.get_states(batch_size)
        assert states.shape == (batch_size, default_config.initial_states, default_config.state_dim)
        # Check that all batches have the same states
        for i in range(1, batch_size):
            assert torch.allclose(states[0], states[i])
            
    def test_update_importance_scores(self, default_config):
        """Test updating importance scores."""
        manager = StateManager(default_config)
        batch_size = 2
        new_scores = torch.rand(batch_size, default_config.initial_states)
        
        manager.update_importance_scores(new_scores)
        
        # Check that scores are updated
        assert torch.allclose(manager.importance_scores, new_scores)
        
    def test_prune_low_importance_states(self, default_config):
        """Test pruning low importance states."""
        manager = StateManager(default_config)
        # Set some importance scores below threshold
        new_scores = torch.ones(1, default_config.initial_states)
        new_scores[0, 0] = 0.1  # Below threshold
        new_scores[0, 1] = 0.2  # Below threshold
        manager.update_importance_scores(new_scores)
        
        num_pruned = manager.prune_low_importance_states()
        
        # Check that correct number of states were pruned
        assert num_pruned == 2
        assert manager.states.shape[1] == default_config.initial_states - 2
        assert manager.importance_scores.shape[1] == default_config.initial_states - 2
        
    def test_allocate_states(self, default_config):
        """Test allocating new states."""
        manager = StateManager(default_config)
        initial_num_states = manager.states.shape[1]
        num_to_allocate = 3
        
        num_allocated = manager.allocate_states(num_to_allocate)
        
        # Check that correct number of states were allocated
        assert num_allocated == num_to_allocate
        assert manager.states.shape[1] == initial_num_states + num_to_allocate
        assert manager.importance_scores.shape[1] == initial_num_states + num_to_allocate
        
        # Check that we can't allocate more than max_states
        num_allocated = manager.allocate_states(default_config.max_states)
        # Should only allocate up to max_states
        assert manager.states.shape[1] == default_config.max_states