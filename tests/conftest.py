# tests/conftest.py

import pytest
import torch
from src.nstm.core.types import NSTMConfig

@pytest.fixture
def default_config():
    """Provide a default NSTMConfig instance."""
    return NSTMConfig(
        state_dim=32,
        token_dim=32,  # Typically token_dim == state_dim
        gate_type='gru',
        num_attention_heads=4,
        max_states=16,
        initial_states=8,
        prune_threshold=0.3
    )

@pytest.fixture
def sample_tokens(default_config):
    """Provide a sample token tensor."""
    batch_size = 2
    seq_len = 10
    return torch.randn(batch_size, seq_len, default_config.token_dim)

@pytest.fixture
def sample_states(default_config):
    """Provide a sample state tensor."""
    batch_size = 2
    return torch.randn(batch_size, default_config.initial_states, default_config.state_dim)