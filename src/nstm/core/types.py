# src/nstm/core/types.py

from typing import NamedTuple
import torch

# Represents a state vector.
State = torch.Tensor

# Represents a token vector.
Token = torch.Tensor

# Represents a sequence of states. (batch_size, num_states, state_dim)
States = torch.Tensor

# Represents a sequence of tokens. (batch_size, seq_len, token_dim)
Tokens = torch.Tensor

# Represents importance scores of states. (batch_size, num_states)
StateImportanceScores = torch.Tensor

# Represents weights for relationships between tokens and states.
# (batch_size, seq_len, num_states)
TokenToStateRoutingWeights = torch.Tensor

class NSTMConfig(NamedTuple):
    """Basic configuration parameters for the NSTM model."""
    state_dim: int
    token_dim: int
    gate_type: str = 'gru'  # 'gru' or 'lstm'
    num_attention_heads: int = 4
    max_states: int = 64
    initial_states: int = 16
    prune_threshold: float = 0.3